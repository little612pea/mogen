import os
import re

# 指定包含 14166 个文本文件的目录路径
directory_path = "/home/hjy/mogen/motion-diffusion-model/dataset/HumanML3D/HumanML3D/texts_humanml"

# 存储所有动作集合
action_set = set()

# 定义动作提取的正则表达式
action_pattern = re.compile(r'([a-zA-Z]+/VERB\s[a-zA-Z]+/NOUN)|([a-zA-Z]+/VERB)')

# 遍历目录中所有 .txt 文件
for filename in os.listdir(directory_path):
    if filename.endswith(".txt"):  # 检查文件是否为 .txt 格式
        with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
            text_content = file.read()
            
            # 查找所有符合条件的动作
            actions = action_pattern.findall(text_content)
            
            # 将动作形式标准化后加入动作集合中
            for action_tuple in actions:
                # 提取非空的动作匹配
                action = [act for act in action_tuple if act]
                if action:
                    action_str = action[0].replace("/", "_")  # 替换 / 符号为 _ 符号，形成动作词汇
                    action_set.add(action_str)

# 输出动作集合到一个新的文本文件
output_file = "extracted_actions.txt"
with open(output_file, 'w', encoding='utf-8') as out_file:
    for action in sorted(action_set):
        out_file.write(action + "\n")

print(f"动作集合提取完成，共提取了 {len(action_set)} 个不同的动作，已保存到 {output_file} 中。")
