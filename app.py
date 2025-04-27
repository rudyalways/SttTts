import os
import threading
import time
import urllib.parse
from typing import Iterator

import gradio as gr
import openai
import uvicorn
import whisper
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fish_audio_sdk import ReferenceAudio, Session, TTSRequest

app = FastAPI()


@app.get("/stream_audio")
def stream_audio(text: str = Query(...), speed: float = Query(1.0)):
    audio_stream = text_to_speech_11labs_as_stream(text, speed)
    if not audio_stream:
        return "Error on generating audio stream"
    return StreamingResponse(audio_stream, media_type="audio/mpeg")


def get_audio_player_html(text, speed):
    text_param = urllib.parse.quote(text)
    return f"""
        <audio id="ai-audio" controls autoplay>
            <source src="http://localhost:8000/stream_audio?text={text_param}&speed={speed}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                window.setTimeout(function() {{
                    var audioElement = document.getElementById('ai-audio');
                    if (audioElement) {{
                        console.log("tttt");
                        audioElement.play().catch(function(error) {{
                            console.log("Audio playback failed:", error);
                        }});
                    }}
                }}, 1000);
            }});
        </script>
    """


# Load environment variables
load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = os.path.join(os.getcwd(), 'recordings')
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize OpenAI client
# client = openai.OpenAI(api_key=oai_key)


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


def generate_response(message):
    """Generate response using OpenAI GPT"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Please respond in Simplified Chinese (简体中文)."},
            {"role": "user", "content": message}
        ]
    )
    return response.choices[0].message.content


def text_to_speech(text):
    """Convert text to speech using OpenAI TTS"""
    timestamp = int(time.time())
    output_filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(RECORDINGS_DIR, output_filename)

    # Generate speech from text
    response = openai.Audio.speech.create(
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

    elevenlabs_client = ElevenLabs(
        api_key=os.getenv('ELEVENLABS_API_KEY'),
    )

    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    print(f"Using voice ID: {voice_id}")

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
        print(f"TTS generation failed: {e}")
        return None  # Or return a path to a default error audio file

    # Check file size
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        print("Generated audio file is empty.")
        return None

    return output_path


def text_to_speech_11labs_as_stream(text: str, speed: float = 1.0) -> Iterator[bytes]:
    elevenlabs_client = ElevenLabs(
        api_key=os.getenv('ELEVENLABS_API_KEY'),
    )

    voice_id = os.getenv("ELEVENLABS_VOICE_ID")
    print(f"Using voice ID: {voice_id}")

    try:
        start_time = time.time()  # Start timing
        result = elevenlabs_client.text_to_speech.convert_as_stream(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        end_time = time.time()  # End timing
        print(f"ElevenLabs convert_as_stream execution time: {(end_time - start_time) * 1000:.0f} ms")
        return result
    except Exception as e:
        print(f"TTS generation failed: {e}")
        return None  # Or return a path to a default error audio file


def process_audio(audio_path, voice_speed=1.0):
    """Main function to process audio input and generate response"""
    if audio_path is None:
        return "Please record audio first.", None, None

    # Step 1: Transcribe the audio
    transcription, saved_recording = transcribe_audio(audio_path)

    # Step 2: Generate response
    response_text = generate_response(transcription)

    # # Step 3: Convert response to speech
    # print(f'before text_to_speech_11labs')
    # response_audio = text_to_speech_11labs_as_stream(response_text, speed=voice_speed)
    # if not response_audio:
    #     return transcription, response_text, None  # Or a default error message/audio

    # Return all results
    return transcription, response_text, get_audio_player_html(response_text, voice_speed)


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
            # audio_output = gr.Audio(label="AI Voice Response", autoplay=True)
            audio_html = gr.HTML()

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
        outputs=[transcription_output, response_output, audio_html]
    )

    # Also process when recording is completed
    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input, voice_speed],
        outputs=[transcription_output, response_output, audio_html]
    )

    # Initialize recordings list on load
    demo.load(fn=update_recordings, outputs=recording_gallery)

# Launch the app
if __name__ == "__main__":
    # demo.launch()

    def run_gradio():
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    gradio_thread = threading.Thread(target=run_gradio)
    gradio_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000)
