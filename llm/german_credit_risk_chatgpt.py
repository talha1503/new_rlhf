import openai
import os
import pandas as pd
import time

openai.api_key = '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'

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=0,
               )
    return response.choices[0].message["content"]

prompt = ""

response = get_completion(prompt)
print(response)