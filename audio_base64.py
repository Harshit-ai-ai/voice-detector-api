import base64

with open("/home/harshit/Downloads/voice_preview_will - poetical & measured.wav", "rb") as f:
    audio_bytes = f.read()

audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

print(audio_base64[:200])  # preview
