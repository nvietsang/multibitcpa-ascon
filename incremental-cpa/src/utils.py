import numpy as np
import os

def HW(v):
    return sum([int(b) for b in bin(v)[2:]])

def bytes_to_u64(v: list):
    n = (v[0] << 56) |\
        (v[1] << 48) |\
        (v[2] << 40) |\
        (v[3] << 32) |\
        (v[4] << 24) |\
        (v[5] << 16) |\
        (v[6] <<  8) |\
        (v[7]      )
    return int(n)

def load_trace_matrix(config, indexes):
    '''
    :param indexes:
    '''
    nt_s = config.nt_s
    nt_e = config.nt_e
    _dtype = config.data_type

    # Load traces
    trace_files = [os.path.join(config.path_to_traces, f) for f in os.listdir(config.path_to_traces) if ".npy" in f]
    trace_files = sorted(trace_files)

    # if nn_s >= nn_e or nt_s >= nt_e: raise ValueError
    # if nn_e > len(trace_files): raise ValueError
    
    T = np.zeros((nt_e-nt_s, len(indexes)), dtype=_dtype)    
    for j, i in enumerate(indexes):
        tr = np.load(trace_files[i], mmap_mode='r')
        T[:,j] = tr[nt_s:nt_e]

    print(f"[INFO] Loaded matrix of traces: ", end="")
    print(f"shape {T.shape}, size {T.size * T.itemsize} B or {T.size * T.itemsize / (10**9):.4f} GB")
    return T