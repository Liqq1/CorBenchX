import time
import re
import os
import json
from json.decoder import JSONDecodeError
from openai import OpenAI


client = OpenAI(api_key="", base_url="https://api.deepseek.com")     # 8964

def read_report(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()  # 去除首尾空白字符
            return content
    except FileNotFoundError:
        raise Exception(f"错误：文件 {file_path} 未找到")
    except Exception as e:
        raise Exception(f"读取文件失败：{str(e)}")
    

def extract_sections(report_content):
    """
    Extract the FINDINGS and IMPRESSION sections from the report content.
    It is assumed that the report contains the keywords 'FINDINGS:' and 'IMPRESSION:'.
    """
    findings = ""
    impression = ""
    
    # Extract the FINDINGS section: from 'FINDINGS:' up to 'IMPRESSION:' or end of content.
    findings_match = re.search(r"FINDINGS:\s*(.*?)(?=IMPRESSION:|$)", report_content, re.DOTALL | re.IGNORECASE)
    if findings_match:
        findings = findings_match.group(1).strip()
    
    # Extract the IMPRESSION section: from 'IMPRESSION:' to the end.
    impression_match = re.search(r"IMPRESSION:\s*(.*)", report_content, re.DOTALL | re.IGNORECASE)
    if impression_match:
        impression = impression_match.group(1).strip()
        
    return findings, impression


def error_report(file_path, findings, impression):

    messages = [{"role": "user", "content": f'''
    Report:
    FINDINGS:
    {findings}

    IMPRESSION:
    {impression}

    ###### Error Report Generation Task ######
    You are a junior clinician reviewing the above diagnostic report. As a junior clinician, you may inadvertently introduce some common errors into the report.
    Your task is to introduce three errors into the report. 
    The error should be randomly selected from the following five categories:

    1. Omission: The omission of relevant words or expressions, which encompasses both deletions and missing words (e.g., "fracture" instead of "no fracture").

    2. Insertion: The unintentional insertion of incorrect words or expressions, including inappropriate words, incorrect word substitutions, insertions, or word confusions (e.g., "abnormal" instead of "normal").

    3. Spelling Errors: Spelling mistakes, including word truncations, likely due to manual text processing by radiologists through typing errors or inaccurate selection of text that is to be removed or edited, avoid change pneumothorax to pnuemothorax.

    4. Side Confusion: Errors involving side or orientation (e.g., "right" instead of "left," "lateral" instead of "medial").

    5. Other Errors: Including mistakes in units of measurement (e.g., "centimeter" vs "millimeter"), and punctuation mistakes.

    ### Output Format
    First, output the modified report with three errors introduced.  
    After the report, clearly identify and explain the introduced errors in the following format:  
    [Original Text]: "XXX"  
    [Revised Text]: "YYY"  
    [Error Type]: (Omission / Insertion / Spelling Error / Side Confusion / Other)
    [Error Description]: e.g., Omission of "XXX" in the FINDINGS section, misspelling XXX as XXX, or insertion of XXX. Do not use "changed", "modified", "revised", or "original report" in the Error Description.

    Ensure that:
    - Three errors are introduced per report.
    - The output remains medically realistic.
    - The formatting is consistent and follows the structure exactly as specified.s
    '''}]
    time1 = time.time()
    
    max_retries = 5          # 最大重试次数
    wait_seconds = 2         # 每次重试前等待的秒数

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages
            )
            # 如果调用成功且没有抛出 JSONDecodeError，跳出重试循环
            break
        except JSONDecodeError as e:
            print(f"[Attempt {attempt+1}] JSONDecodeError encountered: {e}")
            if attempt < max_retries - 1:
                print(f"Waiting {wait_seconds} seconds before retrying...")
                time.sleep(wait_seconds)
            else:
                print("Max retries reached. Raising exception.")
                raise e

    reasoning_content = response.choices[0].message.reasoning_content
    content = response.choices[0].message.content
    # print("Message:", messages)
    print(content)
    # pring("Reasoning Content:", reasoning_content)
    print("\nTime:", time.time()-time1)
    # print("\nReasoning Content:", reasoning_content)

    error_folder = "../../data/mimic_multi_error_report2/"
    os.makedirs(error_folder, exist_ok=True)

    directory = os.path.dirname(file_path)
    last_folder = os.path.basename(directory)
    parent_directory = os.path.dirname(directory)
    second_last_folder = os.path.basename(parent_directory)
    error_folder = os.path.join(error_folder, second_last_folder, last_folder)
    txt_filename = os.path.join(error_folder, file_path.split("/")[-1])
    # print('txt_filename:', txt_filename)

    txt_folder = os.path.dirname(txt_filename)
    # print('txt_folder:', txt_folder)
    os.makedirs(txt_folder, exist_ok=True)

    with open(txt_filename, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    
    folder_path = "../../data/mimic_512/p18/"
    all_entries = os.listdir(folder_path)
    # 过滤出仅为文件夹的条目
    folders = [entry for entry in all_entries if os.path.isdir(os.path.join(folder_path, entry))]
    # 对文件夹名称进行排序以保证顺序一致
    folders.sort()
    selected_folders = folders[600:650]
    for idx, folder in enumerate(selected_folders, start=1):
        print(f"{idx}. {folder}")
    # 遍历并打印选中的文件夹路径
    for folder in selected_folders:
        folder_full_path = os.path.join(folder_path, folder)
        # print(folder_full_path)
        for filename in os.listdir(folder_full_path):
            if filename.endswith(".txt"):  
                file_path = os.path.join(folder_full_path, filename)
                print("processing:", file_path)
                report_content = read_report(file_path)
                findings, impression = extract_sections(report_content)
                if not findings and not impression:
                    print(f"No FINDINGS and IMPRESSION sections found in the report: {file_path}")
                    continue
                error_report(file_path, findings, impression)
        print("Done!", folder_full_path)
    print("Done!")