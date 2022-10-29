import numpy as np

class beamformer():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
    
    def process(self, frame):
        # Initialize

        R = np.zeros(shape = [self.channels, self.channels, self.nfft], dtype = np.complex64)
        R_inv = np.zeros(shape = [self.channels, self.channels, self.nfft], dtype = np.complex64)
        atf = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        w = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        bf_out = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        
        for k in range(0, self.nfft):
            R[:, :, k] = np.outer(frame[:, k], np.conjugate(frame[:, k]))
            eig_val, eig_vec = np.linalg.eig(R[:, :, k])
            max_index = np.argmax(eig_val)
            atf[:, k] = eig_vec[max_index]
            R_inv[:, :, k] = np.linalg.inv(R[:, :, k])

            w[:, k] = np.matmul(np.R_inv[:, :, k], np.conjugate(atf[:, k]))/(np.matmul(atf[:, 1], np.matmul(R_inv, np.conjugate(atf[:, k]))))
            bf_out[:, k] = np.matmul(w[:, k], frame[:, k])
    
        return bf_out    