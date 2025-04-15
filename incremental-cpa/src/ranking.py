from heapq import heapreplace, heappush
from queue import PriorityQueue
import numpy as np


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
