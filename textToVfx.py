import os
import torch
import torchaudio
from diffusers import AudioLDM2Pipeline

def textToVfx(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the pretrained model
    pipe = AudioLDM2Pipeline.from_pretrained(
        "cvssp/audioldm-s",
        torch_dtype=torch.float32  # Use float32 if float16 causes issues
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    for prompt in prompts:
        print(f"🎧 Generating: {prompt}")
        audio = pipe(prompt, num_inference_steps=50).audios[0]
        sample_rate = 16000

        file_name = prompt.replace(" ", "_") + ".wav"
        file_path = os.path.join(output_dir, file_name)

        # Save the generated audio
        torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), sample_rate)