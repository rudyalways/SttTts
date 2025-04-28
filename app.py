import asyncio
import os
import threading
import time
import traceback
import urllib.parse
from typing import Iterator

import gradio as gr
import uvicorn
import websockets
import whisper
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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

app = FastAPI()


audio_websocket: WebSocket = None
message_lines: Iterator[str] = []


def get_message_line() -> str:
    global message_lines
    while True:
        if len(message_lines) == 0:  # Block when there are no elements
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            continue

        # Get and remove first element
        message = message_lines[0]
        message_lines = message_lines[1:]

        # Return the first element
        return message


@app.websocket("/ws/lines")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Websocket for receiving message is connected!")
    start_time = time.time()  # Start timing
    stream = elevenlabs_client.text_to_speech.convert_realtime(
        text=get_message_line(),
        voice_id=voice_id,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    end_time = time.time()  # End timing
    print(f"ElevenLabs convert_as_stream execution time: {(end_time - start_time) * 1000:.0f} ms")
    try:
        while True:
            data = await websocket.receive_text()
            # await line_queue.put(data)
            print(f"Received: {data}")
            global message_lines
            message_lines = list(message_lines) + [data]

            for audio_data in stream:
                print(f"Sending audio chunk of size: {len(audio_data)} bytes")
                if audio_websocket:
                    await audio_websocket.send_bytes(audio_data)

    except Exception as e:
        print(f"2 TTS generation failed: {e}")
        print("Error stack trace:")
        print(traceback.format_exc())
        return None  # Or return a path to a default error audio file
    except WebSocketDisconnect:
        print("WebSocket disconnected")


@app.websocket("/ws/audio")
async def audio_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Websocket for sending audio data is connected!")
    try:
        global audio_websocket
        if audio_websocket is not None:
            try:
                audio_websocket.close()
            except Exception as e:
                print(f"Error on closing websocket: {e}")
        audio_websocket = websocket

        # # Get the text parameter from the initial message
        # data = await websocket.receive_json()
        # text = data.get("text", "")
        # speed = data.get("speed", 1.0)

        # # Generate the audio stream
        # audio_stream = text_to_speech_11labs_as_stream(text, speed)
        # if not audio_stream:
        #     await websocket.send_text("Error generating audio stream")
        #     return

        # # Stream the audio chunks to the client
        # for chunk in audio_stream:
        #     await websocket.send_bytes(chunk)

    except WebSocketDisconnect:
        print("Audio WebSocket disconnected")
    except Exception as e:
        print(f"Error in audio websocket: {e}")


def get_audio_player_html():
    return f"""
        <div id="audio-container">
            <audio id="ai-audio" controls autoplay></audio>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', () => {{
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioElement = document.getElementById('ai-audio');
                
                // Create MediaSource object
                const mediaSource = new MediaSource();
                audioElement.src = URL.createObjectURL(mediaSource);
                
                let sourceBuffer;
                const queue = [];
                let updating = false;
                
                function processQueue() {{
                    if (queue.length > 0 && !updating) {{
                        updating = true;
                        sourceBuffer.appendBuffer(queue.shift());
                    }}
                }}
                
                mediaSource.addEventListener('sourceopen', () => {{
                    sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
                    
                    sourceBuffer.addEventListener('updateend', () => {{
                        updating = false;
                        processQueue();
                    }});
                    
                    // Connect to WebSocket
                    const ws = new WebSocket('ws://localhost:8100/ws/audio');
                    
                    ws.onopen = () => {{
                        console.log('WebSocket connected');
                    }};
                    
                    ws.onmessage = (event) => {{
                        if (event.data instanceof Blob) {{
                            const reader = new FileReader();
                            reader.onload = () => {{
                                const arrayBuffer = reader.result;
                                if (sourceBuffer.updating) {{
                                    queue.push(arrayBuffer);
                                }} else {{
                                    updating = true;
                                    sourceBuffer.appendBuffer(arrayBuffer);
                                }}
                            }};
                            reader.readAsArrayBuffer(event.data);
                        }}
                    }};
                    
                    ws.onerror = (error) => {{
                        console.error('WebSocket error:', error);
                    }};
                    
                    ws.onclose = () => {{
                        console.log('WebSocket connection closed');
                        mediaSource.endOfStream();
                    }};
                }});
                
                // Attempt to play once loaded
                audioElement.addEventListener('canplay', () => {{
                    audioElement.play().catch(error => {{
                        console.error('Playback failed:', error);
                    }});
                }});
            }});
        </script>
    """


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

    # def send_to_websockets(sentence: str):
    #     asyncio.run_coroutine_threadsafe(
    #         line_queue.put(buffer),
    #         asyncio.get_event_loop()
    #     )

    uri = "ws://localhost:8100/ws/lines"
    with websockets.sync.client.connect(uri) as websocket:
        for event in stream:
            if isinstance(event, ResponseTextDeltaEvent):
                # Add the new character to the buffer
                buffer += event.delta.rstrip('\n')

                # Check if the buffer ends with a sentence-ending punctuation
                if any(buffer.endswith(p) for p in ["。", "！", "？", "…", "。", "！", "？", "…"]):
                    # Print the complete sentence and clear the buffer
                    # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {buffer}")

                    # Send the completed sentence to the websocket queue
                    websocket.send(buffer)
                    buffer = ""

        # Print any remaining text in the buffer
        if buffer:
            # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {buffer}")
            websocket.send(buffer)
            buffer = ""


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


def text_to_speech_11labs_as_stream(text: str, speed: float = 1.0) -> Iterator[bytes]:
    try:
        start_time = time.time()  # Start timing
        result = elevenlabs_client.text_to_speech.convert_realtime(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        end_time = time.time()  # End timing
        print(f"ElevenLabs convert_as_stream execution time: {(end_time - start_time) * 1000:.0f} ms")
        return result
    except Exception as e:
        print(f"1 TTS generation failed: {e}")
        print("Error stack trace:")
        print(traceback.format_exc())
        return None  # Or return a path to a default error audio file


def process_audio(audio_path, voice_speed=1.0):
    """Main function to process audio input and generate response"""
    if audio_path is None:
        return "Please record audio first.", None, None

    # Step 1: Transcribe the audio
    transcription, saved_recording = transcribe_audio(audio_path)

    # Step 2: Generate response
    generate_response(transcription)

    return transcription


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
            # Initialize audio_html with the audio player HTML
            audio_html.value = get_audio_player_html()

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
        outputs=[transcription_output]
    )

    # Also process when recording is completed
    audio_input.stop_recording(
        fn=process_audio,
        inputs=[audio_input, voice_speed],
        outputs=[transcription_output]
    )

    # Initialize recordings list on load
    demo.load(fn=update_recordings, outputs=recording_gallery)

# Launch the app
if __name__ == "__main__":
    # demo.launch()

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8100)

    def run_gradio():
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    run_gradio()

    # gradio_thread = threading.Thread(target=run_gradio)
    # gradio_thread.start()
