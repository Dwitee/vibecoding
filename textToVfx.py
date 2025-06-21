import os
import re
import torch
import torchaudio
# from diffusers import AudioLDM2Pipeline
from tangoflux import TangoFluxInference

# def textToVfx(prompts, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     # Load the pretrained model once
#     pipe = AudioLDM2Pipeline.from_pretrained(
#         "cvssp/audioldm-s",
#         torch_dtype=torch.float32
#     )
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"🔧 Using device: {device}")
#     pipe = pipe.to(device)

#     for prompt in prompts:
#         try:
#             print(f"🎧 Generating: {prompt}")
#             audio = pipe(prompt, num_inference_steps=10).audios[0]
#             sample_rate = 16000

#             # Sanitize filename
#             file_name = re.sub(r'[^\w\-_.]', '_', prompt) + ".wav"
#             file_path = os.path.join(output_dir, file_name)

#             # Save the generated audio
#             torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), sample_rate)
#         except Exception as e:
#             print(f"❌ Failed to generate audio for prompt '{prompt}': {e}")

def generateTangoVfx(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model = TangoFluxInference(name='declare-lab/TangoFlux')
    for prompt in prompts:
        try:
            print(f"🎧 [TangoFlux] Generating: {prompt}")
            audio = model.generate(prompt, steps=50, duration=10)

            # Sanitize filename
            file_name = re.sub(r'[^\w\-_.]', '_', prompt) + ".wav"
            file_path = os.path.join(output_dir, file_name)

            # Save the generated audio
            torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), 44100)
        except Exception as e:
            print(f"❌ [TangoFlux] Failed to generate audio for prompt '{prompt}': {e}")