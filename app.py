from flask import Flask, request, jsonify, send_file
from textToVfx import generateTangoVfx
from keywordExtractor import extract_sound_keywords
import os
import re
import json
from google import genai
from google.genai import types

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
        location="europe-west2-c",
    )

    prompt = f"Classify the following story into one of these categories: bedtime, horror, joke, thriller, or unknown.\n\nStory:\n{text}\n\nRespond with only the label."
    chat = genai_client.chats.create(model="gemini-2.0-flash-001")

    try:
        response = chat.send_message(prompt)
        tone = response.text.strip().lower()
        return jsonify({'tone': tone})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)