import re

def remove_pos_tags_and_save(input_file, output_file):
    cleaned_list = []
    # 匹配单词后面的VERB或NOUN标签
    pattern = re.compile(r'(_VERB|_NOUN)$')
    
    
    # 从输入文件读取短语
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()  # 去除行末的换行符和空格
            # 去掉_VERB或_NOUN的标签
            cleaned_phrase = pattern.sub('', line)
            cleaned_list.append(cleaned_phrase)
    
    # 将结果保存到txt文件中
    with open(output_file, 'w', encoding='utf-8') as file:
        for cleaned_phrase in cleaned_list:
            file.write(cleaned_phrase + '\n')

# 指定输入和输出文件名
input_file = 'extracted_actions.txt'  # 旧的txt文件名
output_file = 'cleaned_phrases.txt'  # 输出的txt文件名

# 调用函数进行处理
remove_pos_tags_and_save(input_file, output_file)

print(f'Results have been saved to {output_file}')
