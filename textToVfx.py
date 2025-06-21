def textToVfx(prompts, output_dir="generated_audio"):
    from diffusers import AudioLDMPipeline
    import torch
    import torchaudio
    import os

    # Load model from Hugging Face
    repo_id = "cvssp/audioldm-s-full-v2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AudioLDMPipeline.from_pretrained(repo_id)
    if device == "cuda":
        pipe = pipe.to(device).to(torch.float16)
    else:
        pipe = pipe.to(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save 5-second audio clips
    for prompt in prompts:
        print(f"🎧 Generating: {prompt}")
        output = pipe(
            prompt=prompt,
            num_inference_steps=50,
            audio_length_in_s=5.0  # 5 seconds per clip
        )

        audio = output.audios[0]  # NumPy array
        sample_rate = pipe.unet.sample_rate if hasattr(pipe.unet, "sample_rate") else 16000

        # Save audio
        file_name = prompt.replace(" ", "_") + ".wav"
        file_path = os.path.join(output_dir, file_name)
        torchaudio.save(file_path, torch.tensor(audio).unsqueeze(0), sample_rate)
        print(f"✅ Saved: {file_path}")