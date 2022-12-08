import numpy as np
import time
import multiprocessing
import threading
import os
from Cython.Build import cythonize

import concurrent.futures


class beamformer_multi():
    def __init__(self, nfft, channels):
        self.nfft = nfft
        self.channels = channels
        self.bf_out = np.zeros(shape = [self.nfft], dtype = np.complex)
        self.alpha = np.zeros(shape = [self.nfft], dtype = np.complex)
        self.eps = 1e-16
    
    def process_cython(self, frame):
        #start = time.time()
        
        # init
        #self.frame = frame.T # nfft x channels
        #self.R = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        #self.R_inv = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        #self.atf = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)
        #self.w_temp = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)

        cdef double complex R = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        cdef double complex R_inv = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        cdef double complex atf = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)
        cdef double complex w_temp = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)
        cdef double complex bf_out = np.zeros(shape = [self.nfft])
        cdef double complex alpha = np.zeros(shape = [self.nfft], dtype = np.complex)
        cdef double complex c_frame = frame

        cdef int NFFT  = self.nfft
        cdef int ind

        for ind in range(NFFT):
            bf_out[ind] = task2(ind, c_frame, R, R_inv, atf, w_temp, alpha)
            

        self.bf_out = bf_out
        #print('Time: ' + str(time.time() - start))
        return self.bf_out
    
    def process(self, frame):
        #start = time.time()
        
        # init
        self.frame = frame.T # nfft x channels
        self.R = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        self.R_inv = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        self.atf = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)
        self.w_temp = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)


        for ind in range(0, self.nfft):
                self.bf_out[ind], _ = self.task(ind)

        #print('Time: ' + str(time.time() - start))
        return self.bf_out

    def process2(self, frame):
        #start = time.time()
        
        # init
        self.frame = frame.T # nfft x channels
        self.R = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        self.R_inv = np.zeros(shape = [self.nfft, self.channels, self.channels], dtype = np.complex)
        self.atf = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)
        self.w_temp = np.zeros(shape = [self.nfft, self.channels], dtype = np.complex)


        num_workers = os.cpu_count()

        with multiprocessing.Pool(num_workers) as p:
            for out, ind in p.map(self.task, range(0, self.nfft)):
                self.bf_out[ind] = out

        #print('Time: ' + str(time.time() - start))
        return self.bf_out

    def task(self, k):
        curr_frame = self.frame[k, :]
        self.R[k, :, :] = curr_frame.T * curr_frame # [nfft x channels x channels]
        self.R_inv[k, :, :] = np.linalg.pinv(self.eps + self.R[k, :, :]) # [nfft x channels x channels]
        _, eig_vecs = np.linalg.eigh(np.squeeze(self.R[k, :, :]))
        self.atf[k, :] = eig_vecs[0, :]
                            
        self.w_temp[k, :] = np.matmul(self.R_inv[k, :, :], self.atf[k, :])
        self.alpha[k] = np.matmul(np.conjugate(self.w_temp[k, :]), self.atf[k, :])
        bf_out = np.matmul(self.w_temp[k, :], np.conjugate(curr_frame))/(self.eps + self.alpha[k])

        return bf_out, k

def task2(k, frame, R, R_inv, atf, w_temp, alpha):
    cdef double eps = 1e-9

    curr_frame = frame[k, :]
    R[k, :, :] = curr_frame.T * curr_frame # [nfft x channels x channels]
    R_inv[k, :, :] = np.linalg.pinv(eps + R[k, :, :]) # [nfft x channels x channels]
    _, eig_vecs = np.linalg.eigh(np.squeeze(R[k, :, :]))
    atf[k, :] = eig_vecs[0, :]
                            
    w_temp[k, :] = np.matmul(R_inv[k, :, :], atf[k, :])
    alpha[k] = np.matmul(np.conjugate(w_temp[k, :]), atf[k, :])
    bf_out = np.matmul(w_temp[k, :], np.conjugate(curr_frame))/(eps + alpha[k])

    return bf_out
