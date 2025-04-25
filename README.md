# Realtime Talking Agent

A real-time conversational agent that allows users to speak to an AI assistant and receive audio responses.

## Features

- Web UI for audio recording and playback
- Speech-to-text conversion using OpenAI's Whisper model
- AI response generation using OpenAI's GPT model
- Text-to-speech conversion using OpenAI's TTS model

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```
2. Open your browser and go to `http://localhost:5000`
3. Click the "Start Recording" button and speak
4. The AI will transcribe your speech, generate a response, and speak it back to you

## Usage

1. Click "Start Recording" to begin recording your voice
2. Speak clearly into your microphone
3. Click "Stop Recording" when you're done speaking
4. Wait for the AI to process your speech and respond
5. Listen to the AI's spoken response 