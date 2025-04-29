import os
import time
import asyncio
import numpy as np
from typing import AsyncGenerator, Generator
import websockets
import json
import base64
import traceback
import sys
import logging

import gradio as gr
import openai
import whisper
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from fish_audio_sdk import ReferenceAudio, Session, TTSRequest
from faster_whisper import WhisperModel

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Reduce verbose debug messages from python_multipart
logging.getLogger('python_multipart').setLevel(logging.WARNING)

# Load environment variables
load_dotenv(override=True)
oai_key = os.getenv("OPENAI_API_KEY")
openai.api_key = oai_key
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = os.path.join(os.getcwd(), 'recordings')
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# Initialize Whisper model for Mac (using CPU)
whisper_model = whisper.load_model("base")  # Using base model for faster processing on CPU

class StreamingWhisper:
    def __init__(self, model=whisper_model, chunk_mode="stop"):
        self.model = model
        self.buffer = np.array([], dtype=np.float32)  # Initialize with explicit dtype
        self.target_sample_rate = 16000  # Whisper expects 16kHz
        self.chunk_mode = chunk_mode
        # Set chunk length based on mode
        if chunk_mode == "stop":
            self.chunk_length = None  # Will process entire buffer at once
        else:
            # Extract seconds from mode string (e.g., "1sec" -> 1)
            seconds = float(chunk_mode.replace("sec", ""))
            self.chunk_length = int(seconds * self.target_sample_rate)
        self.min_audio_length = 0.5 * self.target_sample_rate

    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        duration = len(audio) / orig_sr
        time_old = np.linspace(0, duration, len(audio))
        time_new = np.linspace(0, duration, int(len(audio) * target_sr / orig_sr))
        return np.interp(time_new, time_old, audio)

    def add_audio_chunk(self, audio_input) -> Generator[str, None, None]:
        if audio_input is None:
            logger.warning("[Whisper] No audio input received.")
            return

        try:
            # Debug the audio input format
            logger.info(f"[Whisper] Received audio input type: {type(audio_input)}")
            
            if isinstance(audio_input, tuple) and len(audio_input) == 2:
                sample_rate, audio_chunk = audio_input
                logger.info(f"[Whisper] Audio details - Sample rate: {sample_rate}, Shape: {np.shape(audio_chunk)}")
            else:
                logger.warning(f"[Whisper] Invalid audio format: {audio_input}")
                return

            # Convert and normalize audio - explicitly set dtype to float32
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
            if len(audio_chunk.shape) > 1:
                logger.info("[Whisper] Converting stereo to mono")
                audio_chunk = np.mean(audio_chunk, axis=1)

            # Resample if necessary
            if sample_rate != self.target_sample_rate:
                logger.info(f"[Whisper] Resampling from {sample_rate}Hz to {self.target_sample_rate}Hz")
                audio_chunk = self.resample(audio_chunk, sample_rate, self.target_sample_rate)
                # Ensure resampled audio is float32
                audio_chunk = audio_chunk.astype(np.float32)

            # Normalize audio
            max_val = np.abs(audio_chunk).max()
            if max_val > 0:
                audio_chunk = audio_chunk / max_val

            # Append to buffer - ensure buffer is float32
            if len(self.buffer) == 0:
                self.buffer = audio_chunk
            else:
                self.buffer = np.append(self.buffer, audio_chunk).astype(np.float32)
            
            if self.chunk_mode == "stop":
                # Only process when we have the complete audio
                if len(self.buffer) >= self.min_audio_length:
                    try:
                        logger.info("[Whisper] Transcribing complete audio...")
                        # Ensure buffer is float32 before transcribing
                        result = self.model.transcribe(self.buffer.astype(np.float32), language='en')
                        text = result["text"].strip()
                        if text:
                            logger.info(f"[Whisper] Transcription successful: {text}")
                            yield text
                        self.buffer = np.array([], dtype=np.float32)  # Clear buffer after processing
                    except Exception as e:
                        logger.error(f"[Whisper] Transcription error: {str(e)}")
                        traceback.print_exc()
            else:
                # Process buffer in chunks
                while len(self.buffer) >= self.chunk_length:
                    chunk = self.buffer[:self.chunk_length].astype(np.float32)
                    self.buffer = self.buffer[self.chunk_length:]
                    
                    if len(chunk) >= self.min_audio_length:
                        try:
                            logger.info("[Whisper] Transcribing audio chunk...")
                            result = self.model.transcribe(chunk, language='en')
                            text = result["text"].strip()
                            if text:
                                logger.info(f"[Whisper] Transcription successful: {text}")
                                yield text
                        except Exception as e:
                            logger.error(f"[Whisper] Transcription error: {str(e)}")
                            traceback.print_exc()

        except Exception as e:
            logger.error(f"[Whisper] Error processing audio: {str(e)}")
            traceback.print_exc()
            return

class StreamingLLM:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model

    async def generate_response(self, text: str) -> AsyncGenerator[str, None]:
        if not text.strip():
            logger.warning("No transcription provided to LLM.")
            return
        logger.info(f"Sending transcription to LLM: {text}")
        try:
            response_stream = await openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Please respond in Simplified Chinese (简体中文)."},
                    {"role": "user", "content": text}
                ],
                stream=True
            )
            async for chunk in response_stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    logger.info(f"LLM response chunk received.")
                    yield content
        except Exception as e:
            logger.error(f"Error in LLM response generation: {e}")
            return

class StreamingTTS:
    def __init__(self, api_key=ELEVENLABS_API_KEY, voice_id=ELEVENLABS_VOICE_ID):
        self.api_key = api_key
        self.voice_id = voice_id
        self.ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"

    async def stream_tts(self, text: str, chunk_size: int = 1024) -> AsyncGenerator[bytes, None]:
        logger.info("Starting TTS streaming...")
        async with websockets.connect(
            self.ws_url,
            extra_headers={"xi-api-key": self.api_key}
        ) as websocket:
            await websocket.send(json.dumps({
                "text": text,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                },
                "xi_api_key": self.api_key,
            }))
            try:
                while True:
                    response = await websocket.recv()
                    if isinstance(response, str):
                        data = json.loads(response)
                        if data.get("audio"):
                            audio_data = base64.b64decode(data["audio"])
                            logger.info("Received TTS audio chunk.")
                            yield audio_data
                    elif isinstance(response, bytes):
                        logger.info("Received TTS audio bytes chunk.")
                        yield response
            except websockets.exceptions.ConnectionClosed:
                logger.info("TTS WebSocket connection closed.")
                return

async def streaming_pipeline(audio_input, voice_speed: float = 1.0, chunk_mode: str = "stop") -> AsyncGenerator[tuple[str, str, bytes], None]:
    try:
        if audio_input is None:
            logger.warning("[Pipeline] No audio input received")
            return

        logger.info(f"[Pipeline] Starting processing with audio input: {type(audio_input)}")
        logger.info(f"[Pipeline] Chunk mode: {chunk_mode}")
        
        # Initialize components
        whisper_stream = StreamingWhisper(chunk_mode=chunk_mode)
        llm_stream = StreamingLLM()
        tts_stream = StreamingTTS()

        # Process audio through Whisper
        transcription = ""
        try:
            logger.info("[Pipeline] Starting transcription...")
            for text in whisper_stream.add_audio_chunk(audio_input):
                transcription += " " + text
                logger.info(f"[Pipeline] Got transcription: {transcription.strip()}")
                yield transcription.strip(), "", None
        except Exception as e:
            logger.error(f"[Pipeline] Transcription error: {e}")
            traceback.print_exc()
            return

        if not transcription.strip():
            logger.warning("[Pipeline] No transcription generated")
            return

        # Process through LLM
        logger.info("[Pipeline] Starting LLM processing...")
        response_text = ""
        try:
            async for response_chunk in llm_stream.generate_response(transcription):
                response_text += response_chunk
                logger.info(f"[Pipeline] LLM response updated: {response_text}")
                yield transcription, response_text, None
        except Exception as e:
            logger.error(f"[Pipeline] LLM error: {e}")
            return

        # Process through TTS
        logger.info("[Pipeline] Starting TTS processing...")
        try:
            async for audio_chunk in tts_stream.stream_tts(response_text):
                yield transcription, response_text, audio_chunk
        except Exception as e:
            logger.error(f"[Pipeline] TTS error: {e}")
            return

    except Exception as e:
        logger.error(f"[Pipeline] Pipeline error: {e}")
        traceback.print_exc()
        return

# Create Gradio interface
with gr.Blocks(title="Streaming Realtime Talking Agent") as demo:
    gr.Markdown("# Streaming Realtime Talking Agent")
    gr.Markdown("""
    Instructions:
    1. Click 'Record' and speak your message
    2. Click 'Stop' when you finish speaking
    3. Wait for the AI response in text and audio
    
    Chunk Mode:
    - Stop: Process entire audio after stopping (most accurate)
    - 1sec/2sec/3sec: Process audio in chunks while recording (faster response)
    """)

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Record your message (Click 'Stop' after speaking)",
                sources=["microphone"],
                streaming=True,
                type="numpy"
            )
            with gr.Row():
                chunk_mode = gr.Radio(
                    choices=["stop", "1sec", "2sec", "3sec"],
                    value="stop",
                    label="Processing Mode",
                    info="Choose how to process the audio"
                )
                voice_speed = gr.Slider(
                    minimum=0.7,
                    maximum=1.2,
                    value=1.0,
                    step=0.05,
                    label="Voice Speed"
                )

        with gr.Column():
            transcription_output = gr.Textbox(
                label="Your message (transcribed)",
                interactive=False
            )
            response_output = gr.Textbox(
                label="AI Response",
                interactive=False
            )
            audio_output = gr.Audio(
                label="AI Voice Response",
                streaming=True,
                autoplay=True
            )

    # Set up streaming pipeline
    audio_input.stream(
        fn=streaming_pipeline,
        inputs=[audio_input, voice_speed, chunk_mode],
        outputs=[transcription_output, response_output, audio_output],
        show_progress=True,
        queue=True
    )

# Launch the app
if __name__ == "__main__":
    demo.queue(max_size=128).launch(
        debug=True,
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
