from __future__ import print_function
from __future__ import absolute_import
from __future__ import division 


import numpy as np
# import multiprocessing as pmp
import torch.multiprocessing as mp
from collections import defaultdict



class MultiStreamer(object):
    """
    Multithreaded Streaming for Temporally Coherent Batches 
    
    uses the multiprocessing package
    expects the "data" in tensor form with array_dim shape per thread.
    """
    def __init__(self, make_env, array_dim, batchsize, max_q_size, num_threads):
        self.readyQs = [mp.Queue(maxsize=max_q_size) for i in range(num_threads)]
        self.array_dim = array_dim
        self.num_threads = num_threads
        self.num_videos_per_thread = batchsize // num_threads
        self.max_q_size = max_q_size
        self.batchsize = batchsize
        self.make_env = make_env
        self.batch = np.zeros((self.num_threads, self.num_videos_per_thread,
                               *array_dim), dtype=np.float32) 

        array_dim2 = (self.max_q_size, self.num_videos_per_thread,
                      *array_dim)

        self.m_arrays = (mp.Array('f', int(np.prod(array_dim2)), lock=mp.Lock()) for _ in range(num_threads))
        self.arrays = [(m, np.frombuffer(m.get_obj(), dtype='f').reshape(array_dim2)) for m in self.m_arrays]
        self.max_iter = make_env(0).max_rounds

    def multi_stream(self, i, m, n, shape):
        group = self.make_env(num=self.num_videos_per_thread)
        j = 0
        while 1:
            m.acquire()
            info = group.next(n[j])
            self.readyQs[i].put((j, info))
            j = (j+1)%self.max_q_size

    def __iter__(self):
        procs = [mp.Process(target=self.multi_stream, args=(i, m, n, self.array_dim), daemon=True) for i, (m, n) in
                 enumerate(self.arrays)]
        [p.start() for p in procs]
        print('Start Streaming')
        for i in range(self.max_iter):
            batch = defaultdict(list)
            for n in range(self.num_threads):
                j, infos = self.readyQs[n].get() 
                m, arr = self.arrays[n]
                self.batch[n] = arr[j]
                for k, v in infos.items():
                    batch[k] += v
                m.release()
            batch['data'] = self.batch.reshape(self.batchsize, *self.array_dim)
            yield batch
        [p.terminate() for p in procs]