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
from typing import Dict
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import base64
import io
from google.cloud import aiplatform

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
        from google.cloud import aiplatform
        from google.protobuf import json_format
        from google.protobuf.struct_pb2 import Value

        project_id = "secure-garden-460600-u4"
        client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        instance = json_format.ParseDict({"prompt": prompt}, Value())
        instances = [instance]
        parameters = json_format.ParseDict({}, Value())

        endpoint_path = f"projects/{project_id}/locations/us-central1/publishers/google/models/lyria-002"
        print(f"[DEBUG] Calling Vertex AI endpoint: {endpoint_path}")

        response = client.predict(endpoint=endpoint_path, instances=instances, parameters=parameters)
        predictions = response.predictions
        print(f"[DEBUG] Returned {len(predictions)} samples")

        if predictions:
            audio_b64 = predictions[0].get("bytesBase64Encoded")
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                print(f"[DEBUG] Saved background music to temp file: {temp_audio.name}")
                return send_file(temp_audio.name, as_attachment=True)
        else:
            print("[ERROR] No predictions returned from Lyria")
            return jsonify({'error': 'No audio generated'}), 500
    except Exception as e:
        print(f"[ERROR] Exception during background music generation: {e}")
        return jsonify({'error': str(e)}), 500


# === New TTS route ===
@app.route('/generateTTS', methods=['POST'])
def generateTTS():
    data = request.get_json()
    text = data.get('text', '')
    tone = data.get('tone', 'soft')
    voice = data.get('voice', 'female')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    print(f"[DEBUG] Received TTS request with text: {text}, tone: {tone}, voice: {voice}")

    try:
        project_id = "secure-garden-460600-u4"
        model_name = "text-to-speech"
        location = "us-central1"
        api_endpoint = f"{location}-aiplatform.googleapis.com"

        client_options = {"api_endpoint": api_endpoint}
        client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

        prompt = f"Generate a {tone} {voice} voice for the following text: {text}"
        instance = json_format.ParseDict({"text": prompt}, Value())
        instances = [instance]
        parameters = json_format.ParseDict({}, Value())

        endpoint_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_name}"
        print(f"[DEBUG] Calling Vertex AI TTS endpoint: {endpoint_path}")

        response = client.predict(endpoint=endpoint_path, instances=instances, parameters=parameters)
        predictions = response.predictions
        print(f"[DEBUG] Returned {len(predictions)} TTS samples")

        if predictions:
            audio_b64 = predictions[0].get("bytesBase64Encoded")
            audio_bytes = base64.b64decode(audio_b64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_bytes)
                temp_audio.flush()
                print(f"[DEBUG] Saved TTS output to temp file: {temp_audio.name}")
                return send_file(temp_audio.name, as_attachment=True)
        else:
            print("[ERROR] No predictions returned from TTS model")
            return jsonify({'error': 'No audio generated'}), 500
    except Exception as e:
        print(f"[ERROR] Exception during TTS generation: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)