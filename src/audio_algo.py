import numpy as np
import time

class beamformer():
    def __init__(self, nfft, channels):
        self.nfft = 2*nfft - 1
        self.channels = channels
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.complex64)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.complex64)
        self.eps = 1e-16
    
    def process(self, frame):
        frame = frame.T # nfft x channels
        R = np.matmul(np.reshape(frame, [self.nfft, self.channels, 1]), np.reshape(np.conjugate(frame), [self.nfft, 1, self.channels])) # [nfft x channels x channels]
        R_inv = np.linalg.pinv(self.eps + R) # [nfft x channels x channels]
        _, eig_vecs = np.linalg.eigh(R)
        atf = np.squeeze(eig_vecs[:, :, -1])
        w_temp = np.squeeze(np.matmul(R_inv, np.reshape(atf, [self.nfft, self.channels, 1])))
        
        #start = time.time()
        #for k in range(0, self.nfft):
        #    self.alpha[k] = np.matmul(np.conjugate(w_temp[k, :]), atf[k, :])
        #    self.bf_out[k] = np.matmul(w_temp[k, :], np.conjugate(frame[k, :]))/(self.eps + self.alpha[k])

        self.alpha = w_temp[:, 0]*np.conjugate(atf[:, 0]) + w_temp[:, 1]*np.conjugate(atf[:, 1]) + w_temp[:, 2]*np.conjugate(atf[:, 2]) + w_temp[:, 3]*np.conjugate(atf[:, 3])
        self.bf_out = (w_temp[:, 0]*np.conjugate(frame[:, 0]) + w_temp[:, 1]*np.conjugate(frame[:, 1]) + w_temp[:, 2]*np.conjugate(frame[:, 2]) + w_temp[:, 3]*np.conjugate(frame[:, 3]))/(self.eps + self.alpha)

        #print('Time: ' + str(time.time() - start))
        
        return self.bf_out



