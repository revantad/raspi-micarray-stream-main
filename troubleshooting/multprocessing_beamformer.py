import numpy as np
import time
import multiprocessing
import threading
import os

import concurrent.futures


class beamformer_multi():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.complex)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.complex)
        self.eps = 1e-16
    
    def process2(self, frame):
        start = time.time()
        
        # init
        frame = frame.T # nfft x channels
        R = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        R_inv = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        atf = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)
        w_temp = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)


        num_workers = os.cpu_count()
        inp = range(0,10)
        with multiprocessing.Pool(num_workers) as p:
            _ = p.map(self.task(R, R_inv, atf, w_temp, frame), inp)
            p.close()
            p.join()
        print('Time: ' + str(time.time() - start))

        return self.bf_out

    def task(self, R, R_inv, atf, w_temp, frame):
        
        for k in range(0, self.nfft):
            curr_frame = frame[k, :]
            R[k, :, :] = curr_frame.T * curr_frame # [nfft x channels x channels]
            R_inv[k, :, :] = np.linalg.pinv(self.eps + R[k, :, :]) # [nfft x channels x channels]
            _, eig_vecs = np.linalg.eigh(np.squeeze(R[k, :, :]))
            atf[k, :] = eig_vecs[0, :]
                            
            w_temp[k, :] = np.matmul(R_inv[k, :, :], atf[k, :], out = w_temp[k, :])
            self.alpha[k] = np.matmul(np.conjugate(w_temp[k, :]), atf[k, :])
            self.bf_out[k] = np.matmul(w_temp[k, :], np.conjugate(curr_frame))/(self.eps + self.alpha[k])


