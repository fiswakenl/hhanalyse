#!/usr/bin/env python3

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

try:
    response = client.chat.completions.create(
        extra_headers={
            "HTTP-Referer": "https://github.com/hhscribe",
            "X-Title": "HH Scribe Tech Extractor",
        },
        model="openai/gpt-oss-20b:free",
        messages=[
            {
                "role": "user", 
                "content": "Hello! Return just the word 'test' without any explanation."
            }
        ],
        temperature=0.1,
        max_tokens=100
    )
    
    print("Response object:", response)
    print("Choices:", response.choices)
    if response.choices:
        print("Message:", response.choices[0].message)
        if response.choices[0].message:
            print("Content:", response.choices[0].message.content)
    
except Exception as e:
    print(f"Error: {e}")