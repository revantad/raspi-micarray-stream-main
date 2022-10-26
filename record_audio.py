from src.utils import *

chunk = 1024
seconds = 10
RAobject = record_audio(chunk, seconds)
RAobject.recordAudio()
RAobject.stopRecording()
RAobject.saveWave('audio_recordings/test.wav')
