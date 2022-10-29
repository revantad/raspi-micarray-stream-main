import numpy as np
import time

class beamformer():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.float32)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.float32)
        self.eps = 1e-16
    
    def process(self, frame):
        frame = frame.T # nfft x channels
        R = np.matmul(np.reshape(frame, [self.nfft, self.channels, 1]), np.reshape(np.conjugate(frame), [self.nfft, 1, self.channels])) # [nfft x channels x channels]
        R_inv = np.linalg.pinv(self.eps + R) # [nfft x channels x channels]
        eig_vals, eig_vecs = np.linalg.eigh(R)
        atf = np.squeeze(eig_vecs[:, -1, :])
        w_temp = np.squeeze(np.matmul(R_inv, np.reshape(atf, [self.nfft, self.channels, 1])))
        
        start = time.time()
        for k in range(0, self.nfft):
            self.alpha[k] = np.matmul(np.conjugate(w_temp[k, :]), atf[k, :])
            self.bf_out[k] = np.matmul(w_temp[k, :], np.conjugate(frame[k, :]))/self.alpha[k]
        print('Time: ' + str(time.time() - start))
        
        return self.bf_out



