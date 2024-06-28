

import whisper
import os, sys
HERE = os.path.dirname(__file__)

# use base model
model = whisper.load_model("base", download_root=os.path.join(HERE, 'weights'))

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(os.path.join(HERE, "speech1_orig.wav"))
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
# for cpu
cpu = True
if cpu:
    options = whisper.DecodingOptions(fp16 = False)
else:
    options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)