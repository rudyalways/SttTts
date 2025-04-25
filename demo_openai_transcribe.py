# !pip install -q openai
# !pip install pydub
# !pip install dotenv

import openai
from dotenv import load_dotenv
import os
import time

start_total = time.time()

# Load environment variables
start_env = time.time()
load_dotenv()
oai_key = os.getenv("OPENAI_API_KEY")
end_env = time.time()
print(f"Time to load environment: {end_env - start_env:.2f} seconds")

'''
from pydub import AudioSegment 

# Open an mp3 file 
song = AudioSegment.from_file("Trump_WEF_2018.mp3", 
							format="mp3") 

# pydub does things in milliseconds 
ten_mins = 300 * 1000

# song clip of 10 seconds from starting 
first_10_mins = song[:ten_mins] 
last = song[ten_mins:]

# save file 
first_10_mins.export("first_10_mins.mp3", 
						format="mp3") 
last.export("last.mp3", 
			format="mp3")
print("New Audio file is created and saved") 
'''

# Open and read audio file
start_file = time.time()
audio_file = open("/Users/maybewu/Downloads/first_10_mins.mp3", "rb")
end_file = time.time()
print(f"Time to open audio file: {end_file - start_file:.2f} seconds")

# Initialize OpenAI client
start_client = time.time()
client = openai.OpenAI(api_key=oai_key)
end_client = time.time()
print(f"Time to initialize OpenAI client: {end_client - start_client:.2f} seconds")

# Perform transcription
print("\nStarting transcription...")
start_transcribe = time.time()
transcription = client.audio.transcriptions.create(
    model="gpt-4o-transcribe", 
    file=audio_file
)
end_transcribe = time.time()
print(f"Time for API transcription: {end_transcribe - start_transcribe:.2f} seconds")

print("\nTranscription result:")
print(transcription.text)

end_total = time.time()
print(f"\nBreakdown of processing times:")
print(f"- Environment loading: {end_env - start_env:.2f} seconds")
print(f"- File opening: {end_file - start_file:.2f} seconds")
print(f"- Client initialization: {end_client - start_client:.2f} seconds")
print(f"- API transcription: {end_transcribe - start_transcribe:.2f} seconds")
print(f"Total execution time: {end_total - start_total:.2f} seconds")

# Clean up
audio_file.close()
