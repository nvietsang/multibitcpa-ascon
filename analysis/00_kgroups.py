
#
# Determine groups of key candidates whose distributions are highly correlated
#

import numpy as np
import pickle
import argparse
from tqdm import tqdm

def HW(v):
    return sum([int(b) for b in bin(v)[2:]])

#
# z0, z1, z4 yield the same table
#

def f0(nd, k0, n0, n1):
    mask_sb = (1 << (3 * nd)) - 1
    s = k0 & (n1 ^ mask_sb) ^ n0

    mask_li = (1 << nd) - 1
    s0 = (s >> (2 * nd)) & mask_li
    s1 = (s >> (    nd)) & mask_li
    s2 = (s            ) & mask_li
    return HW(s0 ^ s1 ^ s2)

# def f1(nd, k1, n0, n1):
#     mask_sb = (1 << (3 * nd)) - 1
#     s = n0 & (k1 ^ mask_sb) ^ n1

#     mask_li = (1 << nd) - 1
#     s0 = (s >> (2 * nd)) & mask_li
#     s1 = (s >> (    nd)) & mask_li
#     s2 = (s            ) & mask_li
#     return HW(s0 ^ s1 ^ s2)

def main(config):
    nd = config.nd
    nk = 1<<(3*nd)
    nn = 1<<(6*nd)
    mask = (1<<(3*nd)) - 1


    distribution = []
    for k0 in tqdm(range(nk)):
        d = []
        for n in range(nn):
            n0 = (n >> (3*nd)) & mask
            n1 = n & mask        
            v = f0(nd, k0, n0, n1)
            d.append(v)
        distribution.append(d)

    key_groups = [[i] for i in range(nk)]
    for i in range(nk):
        print(f"{i:3d}: ", end="")
        for j in range(i+1, nk):
            cor = np.abs(np.corrcoef(distribution[i], distribution[j])[0,1])
            if cor > 0:
                print(f"{j:2d} ({cor:.02f}) ", end="")
                key_groups[i].append(j)
                key_groups[j].append(i)
        count = len(key_groups[i])
        print(f"> total {count}")

    print(f"Saving to kgroups_{nd}b.pkl")
    with open(f"kgroups_{nd}b.pkl", "wb") as f: pickle.dump(key_groups, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        dest="nd",
                        help="Number of bits for the selection function",
                        required=True,
                        type=int)
    
    config = parser.parse_args()

    main(config)

        
