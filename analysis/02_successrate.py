import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from heapq import heapreplace, heappush
from queue import PriorityQueue

key = 0x000102030405060708090a0b0c0d0e0f
k0 = (key >> 64) & 0xffffffffffffffff
k1 = key & 0xffffffffffffffff

# For nd = 1
indexes_k0b1z0 = [1, 2, 5, 8, 11, 12, 15, 18, 19, 22, 25, 28, 29, 32, 35, 39, 42, 45, 49, 52, 55, 59, 62]     # 23
indexes_k0b1z4 = [1, 6, 7, 9, 11, 12, 17, 22, 23, 25, 27, 28, 33, 34, 38, 39, 43, 44, 48, 49, 54, 55, 59, 60] # 24
indexes_k1b1z1 = [3, 5, 6, 7, 13, 15, 17, 22, 24, 26, 32, 33, 34, 39, 41, 43, 45, 50, 51, 52, 53, 58, 60, 62] # 24

# For nd = 2
indexes_k0b2z0 = [2, 10, 19, 22, 24, 26, 36, 40, 42, 44, 49, 51, 53, 56] # 14
indexes_k0b2z4 = [3, 9, 14, 19, 24, 30, 35, 41, 46, 51, 56, 62]          # 12
indexes_k1b2z1 = [2, 10, 11, 20, 22, 28, 30, 37, 39, 43, 46, 48, 55, 57] # 14

# For nd = 3
indexes_k0b3z0 = [22, 28, 34, 40, 45, 48, 41, 43, 56, 61] # 10
indexes_k0b3z4 = [20, 23, 33, 36, 39, 42, 45, 48, 51, 60] # 10
indexes_k1b3z1 = [1, 7, 13, 20, 29, 32, 39, 48, 58]       # 9



class RankingQueue(PriorityQueue):
    def _put(self, item):
        if len(self.queue) < self.maxsize-1:
            heappush(self.queue, item)
        elif item > self.queue[0]:
            heapreplace(self.queue, item)
        else: pass

def ranking_cpa(rho, nr=5):
    q = RankingQueue(nr+1)
    for k, cors in enumerate(rho): q.put((float((np.nanmax(np.abs(cors)))), k))
    return sorted(q.queue, reverse=True)

def postprocess_guess(config, rho, nd, order=1):
    if not config.boost:
        assert order == 1, "Success order must be 1 without boosting"
        return [np.argmax(rho.max(1).reshape((1<<(3*nd),)))]

    if nd == 1:
        top = np.argsort(rho.max(1).reshape((8,)))[::-1]
        return list(top[:order])
    
    if   nd == 2: keygroups_file = "kgroups_2b.pkl"; nranks = 15
    elif nd == 3: keygroups_file = "kgroups_3b.pkl"; nranks = 169
    else: raise ValueError
    with open(keygroups_file, "rb") as f: key_groups = pickle.load(f)

    nk = 1 << (3*nd)
    ranked = ranking_cpa(rho,nranks)
    scores = np.zeros(nk, dtype=np.uint8)
    for _, candidate in ranked:
        for k, group in enumerate(key_groups):
            if candidate in group:
                scores[k] += 1
    top = np.argsort(scores)[::-1]
    return list(top[:order])

def extract_bit(k, j):
    return (k >> (63-j)) & 1

def trim_tuple(x: int, i0, nd, selection_function):
    '''
    :param x: an integer of 64 bits
    '''
    if   selection_function == "z0":
        i1 = (i0 + 36) % 64
        i2 = (i0 + 45) % 64
    elif selection_function == "z1":
        i1 = (i0 +  3) % 64
        i2 = (i0 + 25) % 64
    elif selection_function == "z4":
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

# Attention for k1
# - The recovered key is k0 ^ k1 (not k1)
# - Do not forget addition of constant
def process_k1(i0, guess, nd):
    i1 = (i0 +  3) % 64
    i2 = (i0 + 25) % 64
    k0tup = 0
    for i in range(nd):
        k0b00 = extract_bit(k0, (i0+i)%64)
        if 56 <= (i0+i)%64 < 60: k0b00 ^= 1
        k0tup |= k0b00 << (3*nd-i-1)

        k0b03 = extract_bit(k0, (i1+i)%64)
        if 56 <= (i1+i)%64 < 60: k0b03 ^= 1
        k0tup |= k0b03 << (2*nd-i-1)
        
        k0b25 = extract_bit(k0, (i2+i)%64)
        if 56 <= (i2+i)%64 < 60: k0b25 ^= 1
        k0tup |= k0b25 << (nd-i-1)

    k1tup = k0tup ^ guess
    return k1tup

def success_rates(config, 
                  target_key, 
                  selection_function, 
                  indexes, 
                  nd,
                  order, 
                  nbtrace_range):
    
    if   target_key == "k0": kref = k0
    elif target_key == "k1": kref = k1
    else: raise ValueError

    subkeys_ref = dict()
    for j in indexes:
        subkey = trim_tuple(kref, j, nd, selection_function)
        subkeys_ref[j] = subkey

    successrates = [0] * len(nbtrace_range)
    count = 0

    path_prefix = f"{config.path_to_checkpoints}/{target_key}/{selection_function}/b{nd}"
    for idx in indexes:
        for rep in os.listdir(f"{path_prefix}/{idx:02d}"):
            if ".DS_Store" in rep: continue
            path_to_rep = f"{path_prefix}/{idx:02d}/{rep}"
            
            for i, nbtr in enumerate(nbtrace_range):
                path_to_file = f"{path_to_rep}/{nbtr:07d}.pkl"
                if not os.path.exists(path_to_file):
                    raise FileNotFoundError

                with open(path_to_file, "rb") as f: cp = pickle.load(f)

                rho = np.abs(cp["rho"])
                best_guesses = postprocess_guess(config, rho, nd, order)
                if target_key == "k1":
                    best_guesses = [process_k1(idx, guess, nd) for guess in best_guesses]                
                successrates[i] += (subkeys_ref[idx] in best_guesses)

            count += 1
    return count, successrates

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path-to-checkpoints', 
                        dest='path_to_checkpoints', 
                        help="Path to the folder of checkpoints",
                        default="../bi32-armv6/checkpoints",
                        type=str)
    parser.add_argument("--n-traces",
                        dest="nn",
                        help="Number of traces",
                        default=10000,
                        type=int)
    parser.add_argument("--step",
                        dest="ns",
                        help="Number of traces for each process",
                        default=200,
                        type=int)
    parser.add_argument("--boost",
                        dest="boost",
                        help="Use helping algorithm for key recovery",
                        action="store_true")
    
    config = parser.parse_args()

    nn = config.nn
    ns = config.ns
    nbtrace_range = list(range(ns, nn+1, ns))
    plt.figure(figsize=(6, 4))

    # d=1
    count_k0b1z4, successrates_k0b1z4 = success_rates(config, "k0", "z4", indexes_k0b1z4, 1, 1, nbtrace_range)
    print(f"k0 b1 z4 ({count_k0b1z4:3d}): {successrates_k0b1z4}")
    count_k1b1z1, successrates_k1b1z1 = success_rates(config, "k1", "z1", indexes_k1b1z1, 1, 1, nbtrace_range)
    print(f"k1 b1 z1 ({count_k1b1z1:3d}): {successrates_k1b1z1}")
    a0 = 14
    a1 = 14
    srk0b1z4 = [(v/count_k0b1z4)**a0 for v in successrates_k0b1z4]
    srk1b1z1 = [(v/count_k1b1z1)**a1 for v in successrates_k1b1z1]
    srb1 = [v0*v1*100 for v0, v1 in zip(srk0b1z4, srk1b1z1)]
    plt.plot(nbtrace_range, srb1, label=r"d=1")
    
    # d=2
    successorder = 1
    count_k0b2z4, successrates_k0b2z4 = success_rates(config, "k0", "z4", indexes_k0b2z4, 2, successorder, nbtrace_range)
    print(f"k0 b2 z4 ({count_k0b2z4:3d}): {successrates_k0b2z4}")
    count_k1b2z1, successrates_k1b2z1 = success_rates(config, "k1", "z1", indexes_k1b2z1, 2, successorder, nbtrace_range)
    print(f"k1 b2 z1 ({count_k1b2z1:3d}): {successrates_k1b2z1}")
    a0 = 12
    a1 = 14
    srk0b2z4 = [(v/count_k0b2z4)**a0 for v in successrates_k0b2z4]
    srk1b2z1 = [(v/count_k1b2z1)**a1 for v in successrates_k1b2z1]
    srb2 = [v0*v1*100 for v0, v1 in zip(srk0b2z4, srk1b2z1)]
    plt.plot(nbtrace_range, srb2, label=r"d=2")
    
    # d=3
    successorder = 1
    count_k0b3z4, successrates_k0b3z4 = success_rates(config, "k0", "z4", indexes_k0b3z4, 3, successorder, nbtrace_range)
    print(f"k0 b3 z4 ({count_k0b3z4:3d}): {successrates_k0b3z4}")
    count_k1b3z1, successrates_k1b3z1 = success_rates(config, "k1", "z1", indexes_k1b3z1, 3, successorder, nbtrace_range)
    print(f"k1 b3 z1 ({count_k1b3z1:3d}): {successrates_k1b3z1}")    
    a0 = 10
    a1 = 9
    srk0b3z4 = [(v/count_k0b3z4)**a0 for v in successrates_k0b3z4]
    srk1b3z1 = [(v/count_k1b3z1)**a1 for v in successrates_k1b3z1]
    srb3 = [v0*v1*100 for v0, v1 in zip(srk0b3z4, srk1b3z1)]
    plt.plot(nbtrace_range, srb3, label=r"d=3")

    # Plot
    plt.xlabel("Number of traces")
    plt.ylabel("Success rate (%)")
    # plt.xlim([0, 14000])
    plt.ylim([-5, 105])
    plt.legend()
    plt.tight_layout()
    plt.show()

