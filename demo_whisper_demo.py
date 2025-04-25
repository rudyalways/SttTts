import whisper
import time

start_total = time.time()

# Load model
start_model = time.time()
model = whisper.load_model("turbo")
end_model = time.time()
print(f"Time to load model: {end_model - start_model:.2f} seconds")

# load audio and pad/trim it to fit 30 seconds
start_audio = time.time()
audio = whisper.load_audio("/Users/maybewu/Downloads/Trump_WEF_2018.mp3")
audio = whisper.pad_or_trim(audio)
end_audio = time.time()
print(f"Time to load and preprocess audio: {end_audio - start_audio:.2f} seconds")

# make log-Mel spectrogram and move to the same device as the model
start_mel = time.time()
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
end_mel = time.time()
print(f"Time to generate mel spectrogram: {end_mel - start_mel:.2f} seconds")

# detect the spoken language
start_lang = time.time()
_, probs = model.detect_language(mel)
end_lang = time.time()
print(f"Time to detect language: {end_lang - start_lang:.2f} seconds")
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
start_decode = time.time()
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)
end_decode = time.time()
print(f"Time to decode audio: {end_decode - start_decode:.2f} seconds")

# print the recognized text
print("\nTranscription:")
print(result.text)

end_total = time.time()
print(f"\nBreakdown of processing times:")
print(f"- Model loading: {end_model - start_model:.2f} seconds")
print(f"- Audio loading and preprocessing: {end_audio - start_audio:.2f} seconds")
print(f"- Mel spectrogram generation: {end_mel - start_mel:.2f} seconds")
print(f"- Language detection: {end_lang - start_lang:.2f} seconds")
print(f"- Audio decoding: {end_decode - start_decode:.2f} seconds")
print(f"Total execution time: {end_total - start_total:.2f} seconds")