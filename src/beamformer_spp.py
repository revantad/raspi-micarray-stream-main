import numpy as np
import time

class beamformer():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.complex64)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.complex64)
        self.eps = 1e-16
        self.delta = 1e-4
        self.Rn = self.delta*np.eye(shape = [self.nfft, self.channels, self.channels], dtype = np.complex64)
        #self.Rx = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex64)
        self.Ry = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex64)
        self.p = np.zeros(shape = [self.nfft], dtype = np.float32)
        self.alpha_p = 0.98
        self.alpha_y = 0.98
        self.alpha_n = 0.98
    
    def process(self, frame):
        frame = frame.T # nfft x channels
        # [nfft x channels x channels] --> This is the spatial correlation of the mic data (in STFT Domain)
        self.Ry = self.alpha_y*np.matmul(np.reshape(frame, [self.nfft, self.channels, 1]), np.reshape(np.conjugate(frame), [self.nfft, 1, self.channels])) + (1 - self.alpha_y)*self.alpha_y
        self.Rx = self.Ry - self.Rn

        # Compute SPP
        Rn_inv = np.linalg.pinv(self.eps + self.Rn) # [nfft x channels x channels] --> This R is the spatial correlation of the mic noise (Rnn in STFT domain)
        gamma = np.matmul(np.reshape(frame, [self.nfft, self.channels, 1]), np.reshape(np.conjugate(frame), [self.nfft, 1, self.channels])) + (1 - alpha_y)*self.alpha_y
        # Apply speech presence probability and compute Rx and Rn
        # Recursively estimate Ry, Rn, and spp


        
        _, eig_vecs = np.linalg.eigh(self.Rx) # --> This R is the spatial correlation of the mic speech (Rxx in STFT domain)
        atf = np.squeeze(eig_vecs[:, :, 0])
        w_temp = np.squeeze(np.matmul(Rn_inv, np.reshape(atf, [self.nfft, self.channels, 1])))
        
        #start = time.time()
        #for k in range(0, self.nfft):
        #    self.alpha[k] = np.matmul(np.conjugate(w_temp[k, :]), atf[k, :])
        #    self.bf_out[k] = np.matmul(w_temp[k, :], np.conjugate(frame[k, :]))/(self.eps + self.alpha[k])

        self.alpha = w_temp[:, 0]*np.conjugate(atf[:, 0]) + w_temp[:, 1]*np.conjugate(atf[:, 1]) + w_temp[:, 2]*np.conjugate(atf[:, 2]) + w_temp[:, 3]*np.conjugate(atf[:, 3])
        self.bf_out = (w_temp[:, 0]*np.conjugate(frame[:, 0]) + w_temp[:, 1]*np.conjugate(frame[:, 1]) + w_temp[:, 2]*np.conjugate(frame[:, 2]) + w_temp[:, 3]*np.conjugate(frame[:, 3]))/(self.eps + self.alpha)

        #print('Time: ' + str(time.time() - start))
        
        return self.bf_out



