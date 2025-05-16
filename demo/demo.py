from openai import OpenAI
import json
import base64
from PIL import Image
import io


client = OpenAI(api_key="sk-it9NUFxpm8HjUE1B42Ad21B2347c449bB1D9C188Ef14D23e", base_url="https://api.zyai.online/v1")


TASK_PROMPT = '''
Find the errors in the following report and correct them, and provide the error type and error description.

Report:
{}

Please only output content strictly according to the format below and there is only one error, do not output multiple errors, do not output other content, the format is:
[Error Type]: Fill in one of the following: "Omission, Insertion, Spelling Error, Side Confusion, Other".
[Error Description]: Describe the identified error.
[Correct report]: Revise the original report and provide the corrected version.
'''

TASK_PROMPT_multierror = '''
Find the errors in the following report and correct them, and provide the error type and error description.

Report:
{}

Please only output content strictly according to the format below and there may exist multiple errors. Do not output other content, the format is:
[Error Type]: Fill in one of the following: "Omission, Insertion, Spelling Error, Side Confusion, Other".
[Error Description]: Describe the identified error.
[Correct report]: Revise the original report and provide the corrected version.
'''

def model_inference(image_data_url, prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        stream=False,
        messages=[
            {"role": "system", "content": "You are a senior clinician reviewing the above diagnostic report."},
            {"role": "user", "content": f"{prompt}"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url}
                    }
                ]
            }
        ]
    )
    return completion.choices[0].message.content


def inference(image_path, prompt):
    # Open and resize image to 336x336
    with Image.open(image_path) as img:
        # Convert image to RGB if it has an alpha channel
        if img.mode in ("RGBA", "LA"):
            img = img.convert("RGB")
        
        # Resize directly to 336x336
        resized_img = img.resize((336, 336), Image.Resampling.LANCZOS)
        
        # Save image to a bytes buffer
        buffer = io.BytesIO()
        resized_img.save(buffer, format="JPEG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Construct the image data URL (base64 format)
    mime_type = "image/jpeg"
    image_data_url = f"data:{mime_type};base64,{image_data}"

    output_text = [model_inference(image_data_url, prompt)]
    return output_text

# single error
image_path = './single_error.jpg'
json_path = './rer_data_part1_new_0427.json'
with open(json_path, 'r') as f:
    data = json.load(f)
wrong_report = data[0]['input_report']
prompt = TASK_PROMPT.format(wrong_report)
output_text = inference(image_path, prompt)
print(output_text)

# multi error
image_path = './multi_error.jpg'
json_path = './mimic_error_report_multi_v3.json'
with open(json_path, 'r') as f:
    data = json.load(f)
wrong_report = data[0]['input_report']
prompt = TASK_PROMPT_multierror.format(wrong_report)
output_text = inference(image_path, prompt)
print(output_text)
