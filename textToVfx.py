from diffusers import AudioLDM2Pipeline
import torch
import torchaudio
import os

# Load model from Hugging Face
repo_id = "cvssp/audioldm2"
pipe = AudioLDM2Pipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cpu")  # or "cpu" if you're not using GPU

# List of sound prompts
prompts = [
    "a hammer hitting a wooden surface",
    "a cat meowing",
    "thunder rumbling",
    "glass shattering",
    "footsteps on dry leaves"
]

# Create output directory
output_dir = "generated_audio"
os.makedirs(output_dir, exist_ok=True)

# Generate and save 5-second audio clips
for prompt in prompts:
    print(f"🎧 Generating: {prompt}")
    output = pipe(
        prompt=prompt,
        num_inference_steps=200,
        audio_length_in_s=5.0  # 5 seconds per clip
    )
    
    audio = output.audios[0]  # NumPy array
    sample_rate = pipe.unet.sample_rate if hasattr(pipe.unet, "sample_rate") else 16000

    # Save audio
    file_name = prompt.replace(" ", "_") + ".wav"
    file_path = os.path.join(output_dir, file_name)
    torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), sample_rate)
    print(f"✅ Saved: {file_path}")