import os
from flask import Flask, request, send_file, jsonify
from textToVfx import textToVfx

@app.route("/generateVfxFromText", methods=["POST"])
def generateVfxFromText():
    data = request.get_json()
    prompts = data.get("prompts", [])
    output_dir = "generated_audio"
    
    # Generate audio from prompts
    textToVfx(prompts, output_dir)
    
    # Return the first generated file as response
    file_name = prompts[0].replace(" ", "_") + ".wav"
    file_path = os.path.join(output_dir, file_name)

    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404