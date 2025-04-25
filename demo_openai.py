

import os
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)





# 1. Load API key from environment

# 2. Send a simple chat completion request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Give me a short poem about the ocean."}
    ],
    temperature=0.7,
    max_tokens=100
)

# 3. Print out the assistant’s reply
assistant_reply = response.choices[0].message.content
print("Assistant:", assistant_reply)
