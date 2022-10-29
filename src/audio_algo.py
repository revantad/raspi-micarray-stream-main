import numpy as np

class beamformer():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
        self.R = np.zeros(shape = [self.channels, self.channels, self.nfft], dtype = np.complex64)
        self.R_inv = np.zeros(shape = [self.channels, self.channels, self.nfft], dtype = np.complex64)
        self.atf = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        self.w = np.zeros(shape = [self.channels, self.nfft], dtype = np.complex64)
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.complex64)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.complex64)

    def process(self, frame):
        # Initialize
        # self.R = np.outer(frame, np.conjugate(frame), axis = 2)
        for k in range(0, self.nfft):
            self.R[:, :, k] = np.outer(frame[:, k], np.conjugate(frame[:, k]), self.R[:, :, k])
            eig_val, eig_vec = np.linalg.eig(self.R[:, :, k])
            max_index = np.argmax(eig_val)
            self.atf[:, k] = eig_vec[max_index]
            self.R_inv[:, :, k] = np.linalg.pinv(self.R[:, :, k])
            #print(np.shape(R), np.shape(R_inv), np.shape(atf))
            temp = np.matmul(self.R_inv[:, :, k], np.conjugate(self.atf[:, k]))
            self.alpha[k] = np.matmul(self.atf[:, k], temp, self.alpha[k])
            #print(alpha)
            self.w[:, k] = np.matmul(self.R_inv[:, :, k], np.conjugate(self.atf[:, k]), self.w[:, k])
            self.w[:, k] = self.w[:, k]/self.alpha[k]
            self.bf_out[k] = np.inner(self.w[:, k], np.conjugate(frame[:, k]), self.bf_out[:, k])
        
        return self.bf_out    