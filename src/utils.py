import pyaudio
import wave
import numpy as np
from scipy.fft import fft
#import soundfile as sf
#import librosa as lb
#import scipy as sc

class record_audio():
    
    def __init__(self, chunk, record_secs):
        self.form_1 = pyaudio.paFloat32 # 16-bit resolution
        self.chans = int(4) # 1 channel
        self.samp_rate = int(48000) # 44.1kHz sampling rate
        self.dev_index = int(1) # device index found by p.get_device_info_by_index(ii)

        self.chunk = int(chunk) # 2^12 samples for buffer
        self.record_secs = record_secs # seconds to record
        
        self.audio = pyaudio.PyAudio() # create pyaudio instantiation
        # create pyaudio stream
        self.stream = self.audio.open(format = self.form_1,rate = self.samp_rate,channels = self.chans, \
                    input_device_index = self.dev_index,input = True, \
                    frames_per_buffer=self.chunk)
    
    def recordAudio(self):
        print("recording")
        #frames = []
        frame2 = []

        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((self.samp_rate/self.chunk)*self.record_secs)):
            data = self.stream.read(self.chunk)
            
            data_float = np.fromstring(data, dtype = np.float)
            if ii == 1:
                ddat = np.fft.ifft(np.fft.fft(data_float, n = int(self.chunk//2 + 1)), n = int(self.chunk))
                print(np.size(ddat), np.size(data_float))
                
            #frame2.append(np.fft.fft(np.fft.fft(data_float, n = int(self.chunk/2 + 1)), n = int(self.chunk)))
            frame2.append(data_float)
            #frames.append(data)

        print("finished recording")
        
        self.frames = frame2
        self.stopRecording()
        return frame2

    def stopRecording(self):
        # stop the stream, close it, and terminate the pyaudio instantiation
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def saveWave(self, name):
        # save the audio frames as .wav file
        wavefile = wave.open(name,'wb')
        wavefile.setnchannels(self.chans)
        wavefile.setsampwidth(self.audio.get_sample_size(self.form_1))
        wavefile.setframerate(self.samp_rate)
        wavefile.writeframes(b''.join(self.frames))
        wavefile.close()