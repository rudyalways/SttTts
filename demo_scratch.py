
from huggingface_hub import InferenceClient

import os

from dotenv import load_dotenv

load_dotenv()

hf_key = os.getenv('HUGGINGFACE_TOKEN')

client = InferenceClient(token=hf_key)




#from streamer import ParlerTTSStreamer
from parler_tts.streamer import ParlerTTSStreamer
from parler_tts import ParlerTTSForConditionalGeneration

from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
import numpy as np
import spaces
import torch
from threading import Thread


device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(device)
torch_dtype = torch.float16 if device != "cpu" else torch.float32

repo_id = "parler-tts/parler_tts_mini_v0.1"

jenny_repo_id = "ylacombe/parler-tts-mini-jenny-30H"
#jenny_repo_id='openai/whisper-base.en'


model = ParlerTTSForConditionalGeneration.from_pretrained(
    jenny_repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
).to(device)

tokenizer = AutoTokenizer.from_pretrained(repo_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

print('model done')


def generate_response(audio):
    print('generate_response start')
    gr.Info("Transcribing Audio", duration=5)
    question = client.automatic_speech_recognition(audio).text
    print('generate_response after question')

    messages = [{"role": "system", "content": ("You are a magic 8 ball."
                                              "Someone will present to you a situation or question and your job "
                                              "is to answer with a cryptic adage or proverb such as "
                                              "'curiosity killed the cat' or 'The early bird gets the worm'."
                                              "Keep your answers short and do not include the phrase 'Magic 8 Ball' in your response. If the question does not make sense or is off-topic, say 'Foolish questions get foolish answers.'"
                                              "For example, 'Magic 8 Ball, should I get a dog?', 'A dog is ready for you but are you ready for the dog?'")},
                {"role": "user", "content": f"Magic 8 Ball please answer this question -  {question}"}]
    
    response = client.chat_completion(messages, max_tokens=64, seed=random.randint(1, 5000),
                                      model="mistralai/Mistral-7B-Instruct-v0.3")
    
    print('generate_response after chat')

    response = response.choices[0].message.content.replace("Magic 8 Ball", "").replace(":", "")
    return response, None, None

@spaces.GPU
def read_response(answer):
    print('read_response start')

    play_steps_in_s = 2.0
    play_steps = int(frame_rate * play_steps_in_s)

    description = "Jenny speaks at an average pace with a calm delivery in a very confined sounding environment with clear audio quality."
    description_tokens = tokenizer(description, return_tensors="pt").to(device)

    streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)
    prompt = tokenizer(answer, return_tensors="pt").to(device)

    print('read_response mid')

    generation_kwargs = dict(
        input_ids=description_tokens.input_ids,
        prompt_input_ids=prompt.input_ids,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )

    set_seed(42)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_audio in streamer:
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds")
        yield answer, numpy_to_mp3(new_audio, sampling_rate=sampling_rate)
    
    print('read_response end')






import gradio as gr

with gr.Blocks() as block:
    gr.HTML(
        f"""
        <h1 style='text-align: center;'> Magic 8 Ball 🎱 </h1>
        <h3 style='text-align: center;'> Ask a question and receive wisdom </h3>
        <p style='text-align: center;'> Powered by <a href="https://github.com/huggingface/parler-tts"> Parler-TTS</a>
        """
    )
    with gr.Group():
        with gr.Row():
            audio_out = gr.Audio(label="Spoken Answer", streaming=True, autoplay=True)
            answer = gr.Textbox(label="Answer")
            state = gr.State()
        with gr.Row():
            audio_in = gr.Audio(label="Speak your question", sources="microphone", type="filepath")

    audio_in.stop_recording(generate_response, audio_in, [state, answer, audio_out])\
        .then(fn=read_response, inputs=state, outputs=[answer, audio_out])

block.launch()
