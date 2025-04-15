import pickle
import numpy as np
import matplotlib.pyplot as plt
from heapq import heapreplace, heappush
from queue import PriorityQueue

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


if __name__ == "__main__":
    path_to_checkpoints = "../bi32-armv6/checkpoints/k0/z4/b2/09/00"

    # Latex table
    nn_cp = 10000
    path_to_1cp = f"{path_to_checkpoints}/{nn_cp:07d}.pkl"
    with open(path_to_1cp, "rb") as f: cp = pickle.load(f)
    k_ref = 0
    rhocp = np.abs(cp['rho'])

    ranking = ranking_cpa(rhocp, nr=64)
    for r, item in enumerate(ranking):
        if r % 16 == 0:
            print(
    '''\\begin{tabular}{|ccc|}
        \hline
        Rank & Key & Corr. \\\\
        \hline''')
        if item[1] == k_ref: print(f"    \\textcolor{{blue}}{{{r+1:2d}}} & \\textcolor{{blue}}{{{item[1]:2d}}} & \\textcolor{{blue}}{{{item[0]:.03f}}} \\\\")
        elif r < 15: print(f"    {r+1:2d} & {item[1]:2d} & {item[0]:.03f} \\\\")
        else: print(f"    \\textcolor{{gray}}{{{r+1:2d}}} & \\textcolor{{gray}}{{{item[1]:2d}}} & \\textcolor{{gray}}{{{item[0]:.03f}}} \\\\")

        if r % 16 == 15:
            print('''\hline''')
            print('''\end{tabular}''')

    # Plot
    nn_s = 200
    nn_e = 10000
    step = 200

    corr = dict()
    for k in range(64): corr[k] = []

    for nn in range(nn_s, nn_e+1, step):
        path_to_1cp = f"{path_to_checkpoints}/{nn:07d}.pkl"
        with open(path_to_1cp, "rb") as f: cp = pickle.load(f)
        rho = np.abs(cp['rho'])
        for k, ct in enumerate(rho):
            corr[k].append(max(ct))

    plt.figure(figsize=(6,4))
    x_axis = range(nn_s,nn_e+1,step)

    plt.plot(x_axis, corr[0], color = "lightgray", label="Wrong keys")
    for k in range(1,64):
        plt.plot(x_axis, corr[k], color = "lightgray")

    plt.plot(x_axis, corr[k_ref], color = "blue", label="Correct key")
    plt.xlabel("Number of traces")
    plt.ylabel("Absolute correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()