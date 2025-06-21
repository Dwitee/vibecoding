import os
import re
import torch
import torchaudio
from diffusers import AudioLDM2Pipeline

def textToVfx(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load the pretrained model once
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm-s",
        torch_dtype=torch.float32
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    pipe = pipe.to(device)

    for prompt in prompts:
        try:
            print(f"🎧 Generating: {prompt}")
            audio = pipe(prompt, num_inference_steps=10).audios[0]
            sample_rate = 16000

            # Sanitize filename
            file_name = re.sub(r'[^\w\-_.]', '_', prompt) + ".wav"
            file_path = os.path.join(output_dir, file_name)

            # Save the generated audio
            torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), sample_rate)
        except Exception as e:
            print(f"❌ Failed to generate audio for prompt '{prompt}': {e}")

def generateTangoVfx(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    processor = TANGOProcessor.from_pretrained("nttcslab/tango")
    model = TANGOForConditionalGeneration.from_pretrained("nttcslab/tango")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 [TANGO] Using device: {device}")
    model = model.to(device)

    for prompt in prompts:
        try:
            print(f"🎧 [TANGO] Generating: {prompt}")
            inputs = processor(text=prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            waveform = model.generate(**inputs)
            audio = waveform.cpu().squeeze().numpy()
            sample_rate = 16000

            # Sanitize filename
            file_name = re.sub(r'[^\w\-_.]', '_', prompt) + ".wav"
            file_path = os.path.join(output_dir, file_name)

            # Save the generated audio
            torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), sample_rate)
        except Exception as e:
            print(f"❌ [TANGO] Failed to generate audio for prompt '{prompt}': {e}")