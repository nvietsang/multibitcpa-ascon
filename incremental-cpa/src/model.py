import numpy as np
from utils import bytes_to_u64, HW

C = 0x96
I = 0x80400c0600000000

#
# Selection function linear layer output with
# fine-tuned S-box functions
#

def z0(config, k0, n0, n1):
    '''
    '''
    nd = config.nd
    mask_sb = (1 << (3 * nd)) - 1
    s = k0 & (n1 ^ mask_sb) ^ n0

    mask_li = (1 << nd) - 1
    s0 = (s >> (2 * nd)) & mask_li
    s1 = (s >> (    nd)) & mask_li
    s2 = (s            ) & mask_li
    return HW(s0 ^ s1 ^ s2)

def z4(config, k0, n0, n1):
    nd = config.nd
    mask_sb = (1 << (3 * nd)) - 1
    s = n1 & (k0 ^ mask_sb) ^ n0

    mask_li = (1 << nd) - 1
    s0 = (s >> (2 * nd)) & mask_li
    s1 = (s >> (    nd)) & mask_li
    s2 = (s            ) & mask_li
    return HW(s0 ^ s1 ^ s2)

def z1(config, k1, n0, n1):
    '''
    '''
    nd = config.nd
    mask_sb = (1 << (3 * nd)) - 1
    s = n0 & (k1 ^ mask_sb) ^ n1

    mask_li = (1 << nd) - 1
    s0 = (s >> (2 * nd)) & mask_li
    s1 = (s >> (    nd)) & mask_li
    s2 = (s            ) & mask_li
    return HW(s0 ^ s1 ^ s2)

def extract_bit(x, j):
    '''
    :param x: an integer of 64 bits
    :param j: index
    '''
    return (x >> (63-j)) & 1

def trim_tuple(config, x: int):
    '''
    :param x: an integer of 64 bits
    '''

    i0 = config.i0
    nd = config.nd
    if   config.selection_function == "z0":
        i1 = (i0 + 36) % 64
        i2 = (i0 + 45) % 64
    elif config.selection_function == "z1":
        i1 = (i0 +  3) % 64
        i2 = (i0 + 25) % 64
    elif config.selection_function == "z4":
        i1 = (i0 + 57) % 64
        i2 = (i0 + 23) % 64
    else: raise ValueError

    tup = 0
    for i in range(nd):
        b00 = extract_bit(x, (i+i0)%64)
        tup |= b00 << (3*nd-i-1)
        b36 = extract_bit(x, (i+i1)%64)
        tup |= b36 << (2*nd-i-1)
        b45 = extract_bit(x, (i+i2)%64)
        tup |= b45 << (nd-i-1)
    return tup

def load_guess_matrix_z(config, indexes):
    '''
    Write down full guess matrix K

    :param indexes:
    '''
    nk = config.nk
    nd = config.nd

    if   config.selection_function == "z0": f = z0
    elif config.selection_function == "z1": f = z1
    elif config.selection_function == "z4": f = z4
    else: raise ValueError
    
    # Load only part of the file by `mmap_mode`
    nonces = np.load(config.path_to_nonces, mmap_mode='r')
    # if nn_s >= nn_e or nn_e > len(nonces): raise ValueError(f"[ERROR] Invalid number of nonces")
    # nonces = nonces[nn_s:nn_e]

    # Matrix of predicted power consumption K.
    # `dtype=_dtype` defines a proper type for elements corresponding to
    # number of distinguisher bits `nd`. Number of guess bits equals to
    # nd * 3.
    if   1 <= nd <= 2: _dtype = np.uint8
    elif 3 <= nd <= 5: _dtype = np.uint16
    else: raise NotImplementedError
   
    K = np.zeros((nk, len(indexes)), dtype=_dtype)
    
    for j, i in enumerate(indexes):
        non  = nonces[i]
        non0 = bytes_to_u64(list(non[:8]))
        non1 = bytes_to_u64(list(non[8:]))

        n0 = trim_tuple(config, non0)
        n1 = trim_tuple(config, non1)

        for guess in range(nk):
            K[guess,j] = f(config, guess, n0, n1)
        
    print(f"[INFO] Constructed guess matrix: shape {K.shape}, size {K.size * K.itemsize} B or {K.size * K.itemsize / (10**9):.4f} GB")    
    return K


#
# Selection function for pure S-box output
#

def y0(k0, k1, n0, n1, iv, c):    
    return (k0&n1) ^ (k0&k1) ^ (k0&c) ^ (k0&iv) ^ k0 ^ k1 ^ n0 ^ c ^ iv

def y1(k0, k1, n0, n1, iv, c):
    return n1 ^ (n0&(k1^c)) ^ (n0&k0) ^ n0 ^ ((k1^c)&k0) ^ (k1^c) ^ k0 ^ iv

def trim_value(config, v):
    i = config.i0
    nd = config.nd
    mask = (1 << nd) - 1
    return (v >> (64 - (i + nd))) & mask

def load_guess_matrix_y(config, indexes):
    nd = config.nd
    nk = (1 << (2*nd))
    mask = (1 << nd) - 1
    iv = trim_value(config, I)

    if   config.selection_function == "y0": y = y0
    elif config.selection_function == "y1": y = y1
    elif config.selection_function == "y4": raise NotImplementedError
    else: raise ValueError
    
    # Load only part of the file by `mmap_mode`
    nonces = np.load(config.path_to_nonces, mmap_mode='r')
    # if nn_s >= nn_e or nn_e > len(nonces): raise ValueError(f"[ERROR] Invalid number of nonces")
    # nonces = nonces[nn_s:nn_e]

    # Matrix of predicted power consumption K.
    # `dtype=_dtype` defines a proper type for elements corresponding to
    # number of distinguisher bits `nd`. Number of guess bits equals to
    # nd * 3.
    if   1 <= nd <= 8: _dtype = np.uint8
    else: raise NotImplementedError
   
    K = np.zeros((nk, len(indexes)), dtype=_dtype)
    
    for j, i in enumerate(indexes):
        non  = nonces[i]
        non0 = bytes_to_u64(list(non[:8]))
        non1 = bytes_to_u64(list(non[8:]))

        n0 = trim_value(config, non0)
        n1 = trim_value(config, non1)

        for guess in range(nk):
            k0 = (guess >> nd) & mask
            k1 = guess & mask
            # TODO correct c?
            K[guess,j] = y(k0, k1, n0, n1, iv, c=0)
        
    print(f"[INFO] Constructed guess matrix: shape {K.shape}, size {K.size * K.itemsize} B or {K.size * K.itemsize / (10**9):.4f} GB")    
    return K