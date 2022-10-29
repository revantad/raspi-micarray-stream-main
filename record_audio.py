import os
from src.utils import *
from src.audio_algo import *

chunk = 512
seconds = 1
audObject = record_audio(chunk, seconds)

start = time.time()
audObject.recordAudio()
audObject.stopRecording()
print('Time ' + str(time.time() - start))
print('Requested Time ' + str(seconds))

if not os.path.dirname('audio_recordings/'):
    os.makedirs('audio_recordings/')

audObject.saveWave('audio_recordings/test.wav')
