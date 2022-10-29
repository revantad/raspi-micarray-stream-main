import os
from src.utils import *
from src.audio_algo import *

chunk = 2048
seconds = 2
audObject = record_audio(chunk, seconds)
audObject.recordAudio()
audObject.stopRecording()

if not os.path.dirname('audio_recordings/'):
    os.makedirs('audio_recordings/')

audObject.saveWave('audio_recordings/test.wav')
