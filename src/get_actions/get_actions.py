import os
import sys
from openai import OpenAI

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "sk-vem64T04jJLfa918A7745524799940E9A2C7528f97D04b96"

# 创建OpenAI客户端
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://free.gpt.ge/v1"
)

# 英文prompt，用于指导模型提取动作并格式化输出
prompt_template = """
You are provided with sentences, and your task is to extract the main actions (verbs) from each sentence.
Return the output in the following format: action#action/VERB action2#action2/VERB ...#0.0#0.0.
Ensure the format is strictly followed, and if no actions are found, respond with None found#None found#0.0#0.0.

Sentence: {sentence}
"""

# 遍历读取TXT文件并提取主要动作
def process_files(input_folder_path, output_folder_path):
    # 确保输出文件夹存在
    os.makedirs(output_folder_path, exist_ok=True)

    for filename in os.listdir(input_folder_path):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, filename)

            with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
                for line in input_file:
                    if not line.strip():
                        continue
                    
                    # 构建prompt
                    curr_prompt = prompt_template.format(sentence=line.strip())
                    print(f'Processing line: {line.strip()}')

                    # 请求大模型进行生成
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": curr_prompt}],
                        model="gpt-3.5-turbo",
                    )

                    # 写入生成的响应到输出文件
                    response = chat_completion.choices[0].message.content.strip()
                    output_file.write(response + '\n')

# 指定输入和输出文件夹路径
input_folder_path = 'path/to/your/input/txt/files'
output_folder_path = 'path/to/your/output/txt/files'

# 执行处理函数
process_files(input_folder_path, output_folder_path)
