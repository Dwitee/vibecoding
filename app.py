from flask import Flask, request, jsonify
from textToVfx import textToVfx
import os

app = Flask(__name__)

@app.route('/generateVfxFromText', methods=['POST'])
def generateVfxFromText():
    data = request.get_json()
    prompts = data.get('prompts', [])
    output_dir = data.get('output_dir', 'generated_audio')

    if not prompts:
        return jsonify({'error': 'No prompts provided'}), 400

    textToVfx(prompts, output_dir)

    generated_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
    return jsonify({'generated_files': generated_files})

if __name__ == '__main__':
    app.run(debug=True)