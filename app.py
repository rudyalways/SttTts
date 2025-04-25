import os
import time
import gradio as gr
import openai
import whisper
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
oai_key = os.getenv("OPENAI_API_KEY")

# Create recordings directory if it doesn't exist
RECORDINGS_DIR = os.path.join(os.getcwd(), 'recordings')
if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# Load Whisper model
whisper_model = whisper.load_model("base")

# Initialize OpenAI client
client = openai.OpenAI(api_key=oai_key)

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
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
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
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    # Save the audio file
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    return output_path


from fish_audio_sdk import Session, TTSRequest, ReferenceAudio
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

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
def text_to_speech_11labs(text: str):

    timestamp = int(time.time())
    output_filename = f"response_{timestamp}.mp3"
    output_path = os.path.join(RECORDINGS_DIR, output_filename)

    elevenlabs_client = ElevenLabs(
        api_key=os.environ.get('ELEVENLABS_API_KEY'),
    )

    audio = elevenlabs_client.text_to_speech.convert(
        voice_id=os.environ.get('ELEVENLABS_VOICE_ID'),
        model_id='eleven_turbo_v2',
        # optimize_streaming_latency="0",
        output_format='mp3_22050_32',
        text=text,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.5,
            style=0.2,
        ),
    )
    # save audio to mp3
    with open(output_path, "wb") as f:
        # Handle the generator returned by ElevenLabs
        if hasattr(audio, '__iter__'):
            for chunk in audio:
                f.write(chunk)
        else:
            f.write(audio)

    return output_path

def process_audio(audio_path):
    """Main function to process audio input and generate response"""
    if audio_path is None:
        return "Please record audio first.", None, None
    
    # Step 1: Transcribe the audio
    transcription, saved_recording = transcribe_audio(audio_path)
    
    # Step 2: Generate response
    response_text = generate_response(transcription)
    
    # Step 3: Convert response to speech
    response_audio = text_to_speech_11labs(response_text)
    
    # Return all results
    return transcription, response_text, response_audio

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
            submit_btn = gr.Button("Process Audio")
        
        with gr.Column():
            transcription_output = gr.Textbox(label="Your message (transcribed)")
            response_output = gr.Textbox(label="AI Response")
            audio_output = gr.Audio(label="AI Voice Response", autoplay=True)
    
    # History section
    with gr.Accordion("Recording History", open=False):
        gr.Markdown("All recordings are saved in the 'recordings' directory.")
        recording_gallery = gr.Files(label="Saved Recordings", file_count="multiple")
        
        def update_recordings():
            files = [os.path.join(RECORDINGS_DIR, f) for f in os.listdir(RECORDINGS_DIR) 
                    if f.endswith('.mp3')]
            return files
        
        refresh_btn = gr.Button("Refresh Recording List")
        refresh_btn.click(fn=update_recordings, outputs=recording_gallery)
    
    # Set up events
    submit_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[transcription_output, response_output, audio_output]
    )
    
    # Also process when recording is completed
    audio_input.stop_recording(
        fn=process_audio,
        inputs=audio_input,
        outputs=[transcription_output, response_output, audio_output]
    )
    
    # Initialize recordings list on load
    demo.load(fn=update_recordings, outputs=recording_gallery)

# Launch the app
if __name__ == "__main__":
    demo.launch() 