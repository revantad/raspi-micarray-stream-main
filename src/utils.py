import pyaudio
import wave
import numpy as np
import time
from src.audio_algo import *

class record_audio():
    
    def __init__(self, chunk, record_secs):
        self.form_1 = pyaudio.paInt16 # 16-bit resolution
        self.chans = int(4) # 1 channel
        self.samp_rate = int(48000) # 44.1kHz sampling rate
        self.dev_index = int(1) # device index found by p.get_device_info_by_index(ii)
        self.chunk = chunk # 2^12 samples for buffer
        self.nfft = self.chunk//2 + 1
        self.bf_channel = int(1)
        self.record_secs = record_secs # seconds to record
        self.num_frames= self.samp_rate*self.record_secs//self.chunk
        
        self.audio = pyaudio.PyAudio() # create pyaudio instantiation
        # create pyaudio stream
        self.stream = self.audio.open(format = self.form_1, rate = self.samp_rate, channels = self.chans, \
                    input_device_index = self.dev_index, input = True, \
                    frames_per_buffer=self.chunk)

        # Initiate beamformer object
        self.bf = beamformer(self.nfft, self.chans)
    
    def recordAudio(self):
        print("recording")
        
        frames = np.zeros(int(self.chans*self.samp_rate*self.record_secs), dtype = np.int16)
        mic_dat = np.zeros(int(self.chans*self.samp_rate*self.record_secs), dtype = np.int16)
        bf_dat = np.zeros(int(self.bf_channel*self.samp_rate*self.record_secs), dtype = np.int16)

        # loop through stream and append audio chunks to frame array
        for ii in range(0, self.num_frames):
            
            data = self.stream.read(self.chunk, exception_on_overflow = True)
            data_float = np.frombuffer(data, dtype = np.float32)
            
            # Convert float data to matrix of size [channels x frame samples]
            mic_frames = np.reshape(data_float, [self.chans, self.chunk])
            mic_analy = np.fft.rfft(mic_frames, axis = 1, n = self.nfft)
            print(np.shape(mic_analy))
            ## Call audio algorithms/pipeline here
            # Dereverb --> Noise Suppress --> Beamformer
            
            #bf_analy = self.bf.process(mic_analy)
            #bf_synth = np.fft.irfft(bf_analy, axis = 0, n = self.chunk)
            #bf_dat[ii*(self.bf_channel*self.chunk):(ii + 1)*(self.bf_channel*self.chunk)] = bf_synth

            mic_synth = np.fft.irfft(mic_analy, axis = 1, n = self.chunk)
            mic_synth_flat = np.reshape(mic_synth, [1, len(data_float)])
                        
            frames[ii*(self.chans*self.chunk):(ii + 1)*(self.chans*self.chunk)] = data_float
            mic_dat[ii*(self.chans*self.chunk):(ii + 1)*(self.chans*self.chunk)] = mic_synth_flat
            
        
        print("finished recording")
        
        self.frames = frames
        self.bf_dat = bf_dat
        self.mic_dat = mic_dat
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
        wavefile.writeframes(self.mic_dat.tobytes())
        wavefile.close()

        wavefile = wave.open('audio_recordings/test_bf.wav','wb')
        wavefile.setnchannels(self.bf_channel)
        wavefile.setsampwidth(self.audio.get_sample_size(self.form_1))
        wavefile.setframerate(self.samp_rate)
        wavefile.writeframes(self.bf_dat.tobytes())
        wavefile.close()