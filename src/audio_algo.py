import numpy as np
import time

class beamformer():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
        #self.R = np.zeros(shape = [self.channels, self.channels, self.nfft], dtype = np.complex64)
        #self.R_inv = np.zeros(shape = [self.channels, self.channels, self.nfft], dtype = np.complex64)
        #self.atf = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        #self.w = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.complex64)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.complex64)

    #def process(self, frame):
    #    # Initialize
    #    for k in range(0, self.nfft):
    #        self.R[:, :, k] = np.outer(frame[:, k], np.conjugate(frame[:, k]), self.R[:, :, k])
    #        eig_val, eig_vec = np.linalg.eigh(self.R[:, :, k])
     #       
     #       self.atf[:, k] = eig_vec[-1]
     #       self.R_inv[:, :, k] = np.linalg.pinv(self.R[:, :, k])
     #       #print(np.shape(R), np.shape(R_inv), np.shape(atf))
      #      temp = np.matmul(self.R_inv[:, :, k], np.conjugate(self.atf[:, k]))
       #     self.alpha[k] = np.matmul(self.atf[:, k], temp)
        #    #print(alpha)
         #   self.w[:, k] = np.matmul(self.R_inv[:, :, k], np.conjugate(self.atf[:, k]), self.w[:, k])
         #   self.w[:, k] = self.w[:, k]/self.alpha[k]
         #   self.bf_out[k] = np.inner(self.w[:, k], np.conjugate(frame[:, k]))
        
        #return self.bf_out    
    
    def process_vec(self, frame):
        frame = np.reshape(self.channels, 1, self. nfft) # [channels x 1 x nfft]
        R = frame*np.transpose(np.conjugate(frame), [1, 0, 2]) # [nfft x channels x channels]
        R_inv = np.linalg.inv(R) # [nfft x channels x channels]
        eig_vals, eig_vecs = np.linalg.eigh(R)
        atf = np.squeeze(eigVecs[:, -1, :])
        w_temp = np.matmul(R_inv, np.reshape(atf, self.nfft, self.channels, 1))
        
        for k in range(0, self.nfft):
            self.alpha[k] = np.matmul(np.conjugate(w_temp[k, :]), w_temp)
            self.bf_out[k] = np.matmul(w_temp[k, :], np.conjugate(frame))/alpha[k]

        
        return bf_out



