import pyaudio
import wave
import scipy as sc
import numpy as np
import soundfile as sf

class record_audio():
    
    def __init__(self, chunk, record_secs):
        self.form_1 = pyaudio.paInt16 # 16-bit resolution
        self.chans = int(4) # 1 channel
        self.samp_rate = int(48000) # 44.1kHz sampling rate
        self.dev_index = int(1) # device index found by p.get_device_info_by_index(ii)

        self.chunk = int(chunk) # 2^12 samples for buffer
        self.record_secs = record_secs # seconds to record
        
        self.wav_output_filename = '/audio_recording/mic_array.wav' # name of .wav file
        self.audio = pyaudio.PyAudio() # create pyaudio instantiation
        # create pyaudio stream
        self.stream = self.audio.open(format = self.form_1,rate = self.samp_rate,channels = self.chans, \
                    input_device_index = self.dev_index,input = True, \
                    frames_per_buffer=self.chunk)
    
    def recordAudio(self):
        print("recording")
        frames = []

        # loop through stream and append audio chunks to frame array
        for ii in range(0,int((self.samp_rate/self.chunk)*self.record_secs)):
            data = self.stream.read(self.chunk)
            frames.append(data)

        print("finished recording")
        self.frames = frames
        self.stopRecording()
        return frames

    def stopRecording(self):
        # stop the stream, close it, and terminate the pyaudio instantiation
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def saveWave(self):
        # save the audio frames as .wav file
        wavefile = wave.open(self.wav_output_filename,'wb')
        wavefile.setnchannels(self.chans)
        wavefile.setsampwidth(self.audio.get_sample_size(form_1))
        wavefile.setframerate(self.samp_rate)
        wavefile.writeframes(b''.join(self.frames))
        wavefile.close()