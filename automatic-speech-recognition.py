from transformers import pipeline
import torchaudio
# Load the ASR pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-base-960h")
# Path to your audio file
audio_file = "path/to/your/example.wav"
# Perform ASR
transcription = asr_pipeline(audio_file)
