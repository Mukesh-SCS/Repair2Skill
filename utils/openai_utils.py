import openai
import base64
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path):
    encoded_image = encode_image(image_path)

    prompt = "Analyze the furniture and list damaged and intact parts."

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Furniture repair analysis."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}},
        ],
        max_tokens=1000,
    )

    return response.choices[0].message.content