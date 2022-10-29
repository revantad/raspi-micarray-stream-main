import pyaudio
import wave
import numpy as np
from scipy.fft import fft
#import soundfile as sf
#import librosa as lb
#import scipy as sc

class record_audio():
    
    def __init__(self, chunk, record_secs):
        self.form_1 = pyaudio.paInt16 # 16-bit resolution
        self.chans = int(4) # 1 channel
        self.samp_rate = int(48000) # 44.1kHz sampling rate
        self.dev_index = int(1) # device index found by p.get_device_info_by_index(ii)

        self.chunk = chunk # 2^12 samples for buffer
        self.record_secs = record_secs # seconds to record
        self.num_frames = self.samp_rate*self.record_secs//self.chunk
        
        self.audio = pyaudio.PyAudio() # create pyaudio instantiation
        # create pyaudio stream
        self.stream = self.audio.open(format = self.form_1, rate = self.samp_rate, channels = self.chans, \
                    input_device_index = self.dev_index, input = True, \
                    frames_per_buffer=self.chunk)
    
    def recordAudio(self):
        print("recording")
        
        #frames = []
        #frames_dat = []

        frames = np.zeros(int(self.chans*self.samp_rate*self.record_secs), dtype = np.int16)
        frames_dat = np.zeros(int(self.chans*self.samp_rate*self.record_secs), dtype = np.int16)
        
        print(len(frames), len(frames_dat), self.num_frames)
        # loop through stream and append audio chunks to frame array
        for ii in range(0, self.num_frames):
            
            data = self.stream.read(self.chunk)
            data_float = np.frombuffer(data, dtype = np.int16)
            
            # Convert float data to matrix of size [channels x frame samples]
            mic_frames = np.reshape(data_float, [self.chans, self.chunk])
            mic_synth = np.fft.fft(mic_frames, axis = 1, n = int(self.chunk//2 + 1))
            mic_analy = np.real(np.fft.ifft(mic_synth, axis = 1, n = int(self.chunk)))
            mic_signals = np.reshape(mic_analy, [1, len(data_float)])
            

            if ii == 1:
                print(np.shape(mic_frames), np.shape(mic_synth), np.shape(mic_analy))
                print(np.shape(mic_signals))
                print(np.shape(frames))

            
            #frames.append(data_float)
            #frames_dat.append(mic_signals)
            frames[ii*(self.chans*self.chunk):(ii + 1)*(self.chans*self.chunk)] = data_float
            frames_dat[ii*(self.chans*self.chunk):(ii + 1)*(self.chans*self.chunk)] = mic_signals
        
        print(len(frames), len(frames_dat))
        print("finished recording")
        
        self.frames = frames
        self.frames_dat = frames_dat
        self.stopRecording()
        

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
        wavefile.writeframes(self.frames.tobytes())
        wavefile.close()

        wavefile = wave.open('audio_recordings/test_dat.wav','wb')
        wavefile.setnchannels(self.chans)
        wavefile.setsampwidth(self.audio.get_sample_size(self.form_1))
        wavefile.setframerate(self.samp_rate)
        wavefile.writeframes(self.frames_dat.tobytes())
        wavefile.close()