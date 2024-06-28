import whisper  
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
audio_path = "opt/000003.wav"
audio = whisper.load_audio(audio_path)  
audio = whisper.pad_or_trim(audio)
 
model = whisper.load_model("tiny",download_root="./whisper_model/")
 
mel = whisper.log_mel_spectrogram(audio).to(model.device)
 
options = whisper.DecodingOptions(beam_size=5)
 
result = whisper.decode(model, mel, options)  
print(result.text)