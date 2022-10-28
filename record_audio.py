import os
os.system('git pull')

from src.utils import *
chunk = 128
seconds = 10
audObject = record_audio(chunk, seconds)
audObject.recordAudio()
audObject.stopRecording()

if not os.path.dirname('audio_recordings/'):
    os.makedirs('audio_recordings/')

audObject.saveWave('audio_recordings/test.wav')
