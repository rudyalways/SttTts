import asyncio
import os
import threading
import time
import traceback
from typing import Iterator, List

import gradio as gr
import whisper
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from fish_audio_sdk import ReferenceAudio, Session, TTSRequest
from openai import OpenAI

# Load environment variables
load_dotenv(override=True)
# openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = os.path.join(os.getcwd(), 'recordings')
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize OpenAI client
# client = openai.OpenAI(api_key=oai_key)
elevenlabs_client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

voice_id = os.getenv("ELEVENLABS_VOICE_ID")
print(f"Using voice ID: {voice_id}")


def transcribe_audio(audio_path):
    """Transcribe audio using Whisper model"""
    # Save a copy of the audio file with timestamp
    timestamp = int(time.time())
    filename = f"recording_{timestamp}.mp3"
    saved_path = os.path.join(RECORDINGS_DIR, filename)

    # Use whisper to transcribe
    result = whisper_model.transcribe(audio_path)
    transcription = result["text"]
    print(result)

    # Save a copy of the recording
    import shutil
    shutil.copy(audio_path, saved_path)

    return transcription, saved_path


message_texts: List[str] = []


gpt_streaming_finished = False


# def get_text() -> Iterator[str]:
#     global message_texts
#     global retried
#     while retried < 6:
#         if len(message_texts) > 0:
#             result = message_texts[0]
#             message_texts = message_texts[1:]
#             print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] yield: {result}")
#             retried = 0
#             yield result
#         else:
#             retried += 1
#             time.sleep(1)


def get_text() -> Iterator[str]:
    global gpt_streaming_finished
    global message_texts

    while gpt_streaming_finished != True or len(message_texts) > 0:
        if len(message_texts) > 0:
            result = message_texts[0]
            message_texts = message_texts[1:]
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] yield: {result}")
            retried = 0
            yield result
        else:
            time.sleep(0.1)


async def generate_response(message):
    """Generate response using OpenAI GPT and convert to speech in real-time"""
    global message_texts

    stream = client.responses.create(
        model="gpt-4o",
        input=[
            {"role": "system", "content": "You are a helpful assistant. Please respond in Simplified Chinese (简体中文)."},
            {"role": "user", "content": message}
        ],
        stream=True
    )
    from openai.types.responses.response_text_delta_event import \
        ResponseTextDeltaEvent

    # Buffer to store characters
    buffer = ""

    for event in stream:
        if isinstance(event, ResponseTextDeltaEvent):
            # Add the new character to the buffer
            buffer += event.delta.rstrip('\n')

        # Check if the buffer ends with a sentence-ending punctuation
        if any(buffer.endswith(p) for p in ["。", "！", "？", "…", "。", "！", "？", "…"]):
            # Process the complete sentence
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {buffer}")

            # Add the sentence to the text queue
            message_texts.append(buffer)

            # Clear the buffer
            buffer = ""

    # Process any remaining text in the buffer
    if buffer:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {buffer}")

        # Add the remaining sentence to the text queue
        message_texts.append(buffer)
        buffer = ""

    global gpt_streaming_finished
    gpt_streaming_finished = True


def text_to_speech(text):
    """Convert text to speech using OpenAI TTS"""
    timestamp = int(time.time())
    output_filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(RECORDINGS_DIR, output_filename)

    # Generate speech from text
    response = client.Audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    # Save the audio file
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path


def text_to_speech_fish(text):
    timestamp = int(time.time())
    output_filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(RECORDINGS_DIR, output_filename)

    fish_api_key = os.getenv("FISH_API_KEY")

    session = Session(fish_api_key)

    # Option 1: Using a reference_id
    with open(output_path, "wb") as f:
        for chunk in session.tts(TTSRequest(
            reference_id="feed236f66e04575bec1f855f9c7e92d",
            text="Hello, world!" + text
        )):
            f.write(chunk)
    return output_path


def text_to_speech_11labs(text: str, speed: float = 1.0):
    print(f'start text_to_speech_11labs')

    timestamp = int(time.time())
    output_filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(RECORDINGS_DIR, output_filename)

    try:
        # audio = elevenlabs_client.text_to_speech.convert(
        #     voice_id=voice_id,
        #     model_id='eleven_turbo_v2',
        #     output_format='mp3_22050_32',
        #     text=text,
        #     voice_settings=VoiceSettings(
        #         stability=0.5,
        #         similarity_boost=0.5,
        #         style=0.2,
        #         speed=speed,
        #     ),
        # )
        audio = elevenlabs_client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        with open(output_path, "wb") as f:
            if hasattr(audio, '__iter__'):
                wrote = False
                for chunk in audio:
                    f.write(chunk)
                    wrote = True
                if not wrote:
                    raise ValueError("No audio data received from ElevenLabs.")
            else:
                if not audio:
                    raise ValueError("No audio data received from ElevenLabs.")
                f.write(audio)
    except Exception as e:
        print(f"3 TTS generation failed: {e}")
        print("Error stack trace:")
        print(traceback.format_exc())
        return None  # Or return a path to a default error audio file

    # Check file size
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print("Generated audio file is empty.")
        return None

    return output_path


def text_to_speech_11labs_as_stream() -> Iterator[bytes]:
    try:
        start_time = time.time()  # Start timing
        result = elevenlabs_client.text_to_speech.convert_realtime(
            text=get_text(),
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )
        end_time = time.time()  # End timing
        print(f"ElevenLabs convert_as_stream execution time: {(end_time - start_time) * 1000:.0f} ms")

        # Yield each chunk instead of returning the result directly
        return result

    except Exception as e:
        print(f"TTS generation failed: {e}")
        print("Error stack trace:")
        print(traceback.format_exc())
        return None


def text_to_speech_11labs_streaming_to_file():
    global message_texts
    """Convert text to speech using ElevenLabs and save to file"""
    timestamp = int(time.time())
    output_filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(RECORDINGS_DIR, output_filename)

    try:
        start_time = time.time()
        result = elevenlabs_client.text_to_speech.convert_realtime(
            text=get_text(),
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            voice_settings=[]
        )

        # Write the streaming audio to a file
        with open(output_path, "wb") as f:
            # Iterate through the generator and write each chunk
            for chunk in result:
                if chunk is not None:
                    f.write(chunk)

        end_time = time.time()
        print(f"ElevenLabs streaming TTS execution time: {(end_time - start_time) * 1000:.0f} ms")

        return output_path
    except Exception as e:
        print(f"TTS streaming generation failed: {e}")
        print("Error stack trace:")
        print(traceback.format_exc())
        return None


def process_audio(audio_path, voice_speed=1.0):
    """Main function to process audio input and generate response"""
    if audio_path is None:
        return "Please record audio first.", None, None

    # Step 1: Transcribe the audio
    transcription, saved_recording = transcribe_audio(audio_path)

    # Step 2: Generate response in a background thread instead of using asyncio
    threading.Thread(target=lambda: asyncio.run(generate_response(transcription)), daemon=True).start()

    return transcription, "", text_to_speech_11labs_streaming_to_file()


# Create Gradio interface
with gr.Blocks(title="Realtime Talking Agent") as demo:
    gr.Markdown("# Realtime Talking Agent")
    gr.Markdown("Speak into the microphone, and the AI will respond with audio.")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Record your message",
                sources=["microphone"],
                type="filepath"
            )
            voice_speed = gr.Slider(
                minimum=0.7,
                maximum=1.2,
                value=1.0,
                step=0.05,
                label="Voice Speed",
                info="Control the speech rate (0.7-1.2)"
            )
            submit_btn = gr.Button("Process Audio")

        with gr.Column():
            transcription_output = gr.Textbox(label="Your message (transcribed)")
            response_output = gr.Textbox(label="AI Response")
            audio_output = gr.Audio(label="AI Voice Response", streaming=True, autoplay=True)

    # History section
    with gr.Accordion("Recording History", open=False):
        gr.Markdown("All recordings are saved in the 'recordings' directory.")
        recording_gallery = gr.Files(label="Saved Recordings", file_count="multiple")

        def update_recordings():
            files = [os.path.join(RECORDINGS_DIR, f) for f in os.listdir(RECORDINGS_DIR) if f.endswith('.mp3')]
            return files

        refresh_btn = gr.Button("Refresh Recording List")
        refresh_btn.click(fn=update_recordings, outputs=recording_gallery)

    # Set up events
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, voice_speed],
        outputs=[transcription_output, response_output, audio_output]
    )

    # Also process when recording is completed
    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input, voice_speed],
        outputs=[transcription_output, response_output, audio_output]
    )

    # Initialize recordings list on load
    demo.load(fn=update_recordings, outputs=recording_gallery)

# Launch the app
if __name__ == "__main__":
    # demo.launch()

    def run_gradio():
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    run_gradio()

    # gradio_thread = threading.Thread(target=run_gradio)
    # gradio_thread.start()
