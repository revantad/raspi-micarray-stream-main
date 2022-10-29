import os
from src.utils import *
from src.audio_algo import *

chunk = 1024
seconds = 2
audObject = record_audio(chunk, seconds)
start = time.time()
audObject.recordAudio()
audObject.stopRecording()
print('Time ' + str(time.time() - start))
print('Requested Time ' + str(seconds))

if not os.path.dirname('audio_recordings/'):
    os.makedirs('audio_recordings/')

audObject.saveWave('audio_recordings/test.wav')
