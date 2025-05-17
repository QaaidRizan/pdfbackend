from flask import Flask, request, jsonify, Blueprint
import PyPDF2
import os
import re
import sys
import requests
from werkzeug.utils import secure_filename
from uuid import uuid4
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenAI/DeepSeek configuration
OPENROUTER_API_KEY = os.getenv('VITE_OPENROUTER_API_KEY')
SITE_URL = os.getenv('VITE_SITE_URL')
SITE_NAME = os.getenv('VITE_SITE_NAME')

if not OPENROUTER_API_KEY:
    raise ValueError("VITE_OPENROUTER_API_KEY is not set")

# Temporary storage
uploaded_files = {}


# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('\n\n', '\n')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'- (?=[a-z])', '', text)
    text = text.replace('\u2022', '• ')
    text = text.replace('\u00a0', ' ')
    return text.strip()


def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        return clean_text(text)
    except Exception as e:
        return f"Error extracting text: {str(e)}"


# API endpoints
@app.route('/api/upload-pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Instead of saving the file, read it into memory
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        extracted_text = clean_text(text)
        file_id = str(uuid4())
        uploaded_files[file_id] = {
            'filename': filename,
            'text': extracted_text
        }

        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'text_length': len(extracted_text),
            'text_preview': extracted_text[:500] + '...' if len(extracted_text) > 500 else extracted_text
        })

    return jsonify({'error': 'File type not allowed'}), 400



@app.route('/api/query-file', methods=['POST'])
def query_pdf():
    data = request.json
    if not data or 'file_id' not in data or 'prompt' not in data:
        return jsonify({'error': 'file_id and prompt are required'}), 400

    file_id = data['file_id']
    prompt = data['prompt']
    extracted_text = uploaded_files.get(file_id, {}).get('text', None)

    if not extracted_text:
        return jsonify({'error': 'Invalid file_id or no PDF content found'}), 400

    # AI processing
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": SITE_URL,
        "X-Title": SITE_NAME,
        "Content-Type": "application/json"
    }

    prompt_text = f"I'm analyzing a document with the following content:\n\n{extracted_text[:4000]}...\n\nBased on this document, please answer: {prompt}"

    payload = {
        "model": "deepseek/deepseek-v3-base:free",
        "messages": [
            {"role": "system",
             "content": "You are a helpful medical assistant specialized in analyzing brain tumor medical reports. When a user provides a report, read the content carefully and explain the findings clearly and concisely. Present your explanation using numbered points. Focus on helping the user understand the medical terminology, diagnoses, and implications in simple language."
    },
            {"role": "user", "content": prompt_text}
        ]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        ai_response = result["choices"][0]["message"]["content"]
    except Exception as e:
        ai_response = f"Error getting AI response: {str(e)}"

    return jsonify({
        'file_id': file_id,
        'prompt': prompt,
        'response': ai_response
    })


@app.route('/api/test', methods=['GET'])
def test_endpoint():
    return jsonify({'status': 'API is working!'})


# CLI functionality
def run_cli_mode():
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        if len(sys.argv) < 4:
            print("Usage: python app.py --cli <pdf_path> <query>")
            sys.exit(1)

        pdf_path = sys.argv[2]
        query = ' '.join(sys.argv[3:])

        print(f"Processing PDF: {pdf_path}")
        extracted_text = extract_text_from_pdf(pdf_path)
        if extracted_text.startswith("Error"):
            print(extracted_text)
            sys.exit(1)

        print("\nGetting AI response...")
        # Direct AI call
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": SITE_URL,
            "X-Title": SITE_NAME,
            "Content-Type": "application/json"
        }
        prompt_text = f"I'm analyzing a document with the following content:\n\n{extracted_text[:4000]}...\n\nBased on this document, please answer: {query}"
        payload = {
            "model": "deepseek/deepseek-v3-base:free",
            "messages": [
                {"role": "system",
                 "content": "You are an assistant that helps users understand their medical reports. When a user uploads a PDF of a medical report, extract the text and explain the contents of the report in simple, clear language."},
                {"role": "user", "content": prompt_text}
            ]
        }

        try:
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            ai_response = result["choices"][0]["message"]["content"]
        except Exception as e:
            ai_response = f"Error getting AI response: {str(e)}"

        print("\nAI Response:")
        print(ai_response)
        sys.exit(0)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # get PORT from Railway
    app.run(debug=True, host="0.0.0.0", port=port)

