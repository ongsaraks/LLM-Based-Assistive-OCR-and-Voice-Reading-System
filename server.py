from flask import Flask, request, jsonify
import cv2
import numpy as np
from gtts import gTTS
from playsound import playsound
import uuid
import os
import re
from dotenv import load_dotenv
import requests
import json
from openai import OpenAI

# === CONFIG ===
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TEMP_DIR = "temp"
IMG_DIR = "img"

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)
app = Flask(__name__)

# === LABEL SUMMARY CONFIG ===
_SYSTEM_PROMPT = """
Convert OCR text from labels/packages/signs/short docs into clear speech for blind/low-vision users.
Rules:
- if language is not Thai or English strip it out. 
- Keep ALL factual details; do not omit.
- If there is a list, read every item.
- Include all dates, warnings, limits, and constraints.
- You may reorder for clarity, but preserve meaning.
- Do not add or infer anything.
Output: 1–3 short, natural sentences for TTS.
"""

_FEW_SHOT = [
    {
        "role": "user",
        "content": "Ingredients: Sugar, Milk Powder, Cocoa Butter. Best before 12/2026."
    },
    {
        "role": "assistant",
        "content": "ส่วนผสมประกอบด้วยน้ำตาล นมผง และโกโก้บัตเตอร์ วันหมดอายุเดือนธันวาคม ปี 2026"
    },
    {
        "role": "user",
        "content": "ยาแก้ปวด พาราเซตามอล 500 มิลลิกรัม ใช้บรรเทาอาการปวดและลดไข้ ห้ามรับประทานเกินวันละ 8 เม็ด"
    },
    {
        "role": "assistant",
        "content": "ยาพาราเซตามอลขนาด 500 มิลลิกรัม ใช้บรรเทาอาการปวดและลดไข้ จำกัดการรับประทานไม่เกินวันละ 8 เม็ด"
    }
]
load_dotenv()
client = OpenAI(
    api_key=os.getenv('API_KEY'),
    base_url="https://api.opentyphoon.ai/v1"
)

def generate_label(messages, *, stream=False):
    """Generate a short, TTS-friendly label summary from OCR text.

    Args:
        messages: OpenAI chat messages list.
        stream: If True, prints streamed tokens to stdout and also returns the full text.

    Returns:
        Generated text (str).
    """
    if stream:
        response_stream = client.chat.completions.create(
            model="typhoon-v2.5-30b-a3b-instruct",
            messages=messages,
            temperature=0.6,
            max_completion_tokens=512,
            top_p=0.6,
            frequency_penalty=0,
            stream=True,
        )
        chunks = []
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta and delta.content is not None:
                chunks.append(delta.content)
                print(delta.content, end="", flush=True)
        print()
        return "".join(chunks).strip()

    response = client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=512,
        top_p=0.6,
        frequency_penalty=0,
        stream=False,
    )
    content = response.choices[0].message.content
    return (content or "").strip()


def build_label_messages(text: str):
    return (
        [{"role": "system", "content": _SYSTEM_PROMPT}]
        + _FEW_SHOT
        + [{"role": "user", "content": text}]
    )

# @app.route("/label", methods=["POST"])
# def label_summary():
#     payload = request.get_json(silent=True) or {}
#     text = payload.get("text", "")
#     stream = bool(payload.get("stream", False))

#     if not isinstance(text, str) or not text.strip():
#         return jsonify({"error": "no text"}), 400

#     messages = build_label_messages(text)
#     result = generate_label(messages, stream=stream)

#     return jsonify({"text": result})

           

def extract_text_from_image(image_path, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages=None):
    url = "https://api.opentyphoon.ai/v1/ocr"

    with open(image_path, 'rb') as file:
        files = {'file': file}
        data = {
            'model': model,
            'task_type': task_type,
            'max_tokens': str(max_tokens),
            'temperature': str(temperature),
            'top_p': str(top_p),
            'repetition_penalty': str(repetition_penalty)
        }

        if pages:
            data['pages'] = json.dumps(pages)

        headers = {
            'Authorization': f'Bearer {api_key}'
        }

        response = requests.post(url, files=files, data=data, headers=headers)

        if response.status_code == 200:
            result = response.json()

            # Extract text from successful results
            extracted_texts = []
            for page_result in result.get('results', []):
                if page_result.get('success') and page_result.get('message'):
                    content = page_result['message']['choices'][0]['message']['content']
                    try:
                        # Try to parse as JSON if it's structured output
                        parsed_content = json.loads(content)
                        text = parsed_content.get('natural_text', content)
                    except json.JSONDecodeError:
                        text = content
                    extracted_texts.append(text)
                elif not page_result.get('success'):
                    print(f"Error processing {page_result.get('filename', 'unknown')}: {page_result.get('error', 'Unknown error')}")

            return '\n'.join(extracted_texts)
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None



# Usage
def speak(text):
    filename = f"{TEMP_DIR}/{uuid.uuid4()}.mp3"
    tts = gTTS(text=text,lang='th')
    tts.save(filename)
    playsound(filename)
    os.remove(filename)
    
@app.route("/", methods=["GET"])
def index():
    return "OCR and TTS Server is running."

@app.route("/ocr", methods=["POST"])
def ocr_image():
    if "image" not in request.files:
        return jsonify({"error": "no image"}), 400

    img_bytes = request.files["image"].read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # === Preprocess (สำคัญ) ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    #=== Save processed image temporarily ===
    image_id = str(uuid.uuid4())
    processed_path = os.path.join(IMG_DIR, f"{image_id}.png")
    cv2.imwrite(processed_path, thresh)

    # OCR ไทย + อังกฤ
    api_key = os.getenv('API_KEY')
    image_path = processed_path  
    model = "typhoon-ocr"
    task_type = "default"
    max_tokens = 16384
    temperature = 0.1
    top_p = 0.6
    repetition_penalty = 1.2
    pages = None
    
    extracted_text = extract_text_from_image(processed_path, api_key, model, task_type, max_tokens, temperature, top_p, repetition_penalty, pages)
    if extracted_text == "":
        extracted_text = "ไม่พบข้อความ"
    print("OCR Result:", extracted_text)
    messages = build_label_messages(extracted_text)
    result = generate_label(messages)
    print(result)

    # Speak with matched voice
    # if isinstance(summary, str) and summary.startswith("ไม่สามารถสร้างสรุปได้"):
    #     speak(extracted_text)
    # else:
        # speak(summary)
    speak(result)

    return jsonify({
        "text": extracted_text
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
