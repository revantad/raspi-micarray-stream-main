from src.utils import *
import os

chunk = 1024
seconds = 10
RAobject = record_audio(chunk, seconds)
RAobject.recordAudio()
RAobject.stopRecording()

if not os.path.dirname('audio_recordings'):
    os.makedirs('audio_recordings/')

RAobject.saveWave('audio_recordings/test.wav')
