# vibecoding
hackathon for encode  vibe coding 

 an AI-powered immersive storytelling platform that transforms any short story into an expressive audio experience—complete with narrated speech, emotion-aware background music, and synchronized VFX sound effects triggered by story keywords.
What it does:
Classifies the story tone using Gemini Flash (Vertex AI)


Extracts sound-worthy keywords via NLTK NLP


Generates 10-sec mood-based background music with lyria-002 (Vertex AI Media Studio)


Synthesizes expressive narration using nari-labs/dia-1.6b (Vertex AI Model Garden)


Syncs contextual VFX using TangoFlux (Hugging Face) or local .wav assets


Presented in an intuitive Gradio frontend with real-time playback controls

