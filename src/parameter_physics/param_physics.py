from openai import OpenAI
import numpy as np
import torch
import json
import os
import sys

os.environ["OPENAI_API_KEY"] = "sk-vem64T04jJLfa918A7745524799940E9A2C7528f97D04b96"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate(task):
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url="https://free.gpt.ge/v1"
    )

    with open("prompt_parameter_physics.txt", 'r', encoding='utf-8') as file:
        prompt = file.read()
    # define query task
    # task = 'rush to the bathroom with a sense of urgency'
    # task = 'a person doing a deep breathing exercise'
    # find most relevant example
    # construct initial prompt
    curr_prompt = f'{prompt}{task}'
    print(f'\nTask: {task}')

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": curr_prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

    print(chat_completion.choices[0].message.content)
    with open('response.txt', 'w') as file:
        file.write(chat_completion.choices[0].message.content)
if len(sys.argv) > 1:
    # 打印完整的参数(包含空格的字符串)
    print(" ".join(sys.argv[1:]))
generate(" ".join(sys.argv[1:]))