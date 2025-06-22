from flask import Flask, request, jsonify, send_file
from textToVfx import generateTangoVfx
from keywordExtractor import extract_sound_keywords
import os
import re
import json
from google import genai
from google.genai import types
import tempfile
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel

app = Flask(__name__)

@app.route('/generateKeywordsFromText', methods=['POST'])
def generateKeywordsFromText():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    keywords = extract_sound_keywords(text)
    return jsonify({'keywords': keywords})

@app.route('/generateVfxFromText', methods=['POST'])
def generateVfxFromText():
    data = request.get_json()
    prompts = data.get('prompts', [])
    output_dir = data.get('output_dir', 'generated_audio')

    if not prompts:
        return jsonify({'error': 'No prompts provided'}), 400

    generateTangoVfx(prompts, output_dir)

    generated_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    if not generated_files:
        return jsonify({'error': 'No audio generated'}), 500

    first_file_path = os.path.join(output_dir, generated_files[0])
    return send_file(first_file_path, as_attachment=True)

@app.route('/classifyStoryTone', methods=['POST'])
def classifyStoryTone():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    genai_client = genai.Client(
        vertexai=True,
        project="secure-garden-460600-u4",
        location="us-east4",
    )

    prompt = f"Classify the following story into one of these categories: bedtime, horror, joke, thriller, or unknown.\n\nStory:\n{text}\n\nRespond with only the label."
    chat = genai_client.chats.create(model="gemini-2.0-flash-001")

    try:
        response = chat.send_message(prompt)
        tone = response.text.strip().lower()
        return jsonify({'tone': tone})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generateBackgroundMusic', methods=['POST'])
def generateBackgroundMusic():
    data = request.get_json()
    mood = data.get('mood', 'calm')
    duration = data.get('duration', 10)  # duration in seconds

    print(f"[DEBUG] Received request for mood: {mood}, duration: {duration}")

    genai_client = genai.Client(
        vertexai=True,
        project="secure-garden-460600-u4",
        location="us-east4",
    )

    prompt = f"Create a {duration}-second instrumental background music with a {mood} mood."
    print(f"[DEBUG] Generated prompt for Lyria: {prompt}")

    try:
        vertexai.init(project="secure-garden-460600-u4", location="us-east4")
        model = GenerativeModel("models/lyria")
        response = model.generate_content(prompt)
        print(f"[DEBUG] Lyria model response received.")

        if hasattr(response, 'candidates') and response.candidates:
            audio_part = response.candidates[0].content.parts[0]
            if hasattr(audio_part, 'file_data') and hasattr(audio_part.file_data, 'file_uri'):
                uri = audio_part.file_data.file_uri
                print(f"[DEBUG] Received GCS URI: {uri}")
                bucket_name, blob_name = uri.replace("gs://", "").split("/", 1)

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                    print(f"[DEBUG] Downloading audio file to: {temp_audio.name}")
                    blob.download_to_filename(temp_audio.name)
                    print(f"[DEBUG] Audio file downloaded successfully.")
                    return send_file(temp_audio.name, as_attachment=True)
            else:
                print("[ERROR] No file URI in audio part.")
                return jsonify({'error': 'No downloadable URI found in audio data'}), 500
        else:
            print("[ERROR] No candidates in Lyria response.")
            return jsonify({'error': 'No audio generated'}), 500
    except Exception as e:
        print(f"[ERROR] Exception during background music generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)