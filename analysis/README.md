# Analysis

This folder contains python scripts for the analyses in the paper.

## Group for each key candidate

```sh
python3 00_kgroups.py -d 2
```
where $d=2$ is the number of bits for the selection function.

## Optimal number of CPA runs

This requires to install [pycryptosat](https://pypi.org/project/pycryptosat/).

For the bits of $k_0$ and using $z_4$ as the attack point:

```sh
python3 01_nbcparuns.py 64 57 23 14 2
```
where 
- 64 indicates the indexes from 0 to 63, 
- 57 and 23 are the two indexes in the first tuple (0, 57, 23), the index 0 is by default,
- 14 is the upper bound for the number of CPA runs. If we replace 14 with a smaller number, say 13, it returns UNSAT. Therefore, 14 is the optimal number.
- 2 is the number of bits for the selection function

Similarly, for the bits of $k_1$ and using $z_1$ as the attack point:

```sh
python3 01_nbcparuns.py 64 3 25 14 2
```

## Success rates

The CPA must be run before calculating the success rates (see [here](../incremental-cpa/)).

To calculate and plot the success rates for $d=1$, $d=2$ and $d=3$ without using the helping algorithm:

```sh
python3 02_successrate.py
```

Then, with using the helping algorithm:

```sh
python3 02_successrate.py --boost
```

## Second-order success rates

For $d=2$:

```sh
python3 03_successorder2.py
```

For $d=3$:

```sh
python3 04_successorder3.py
```

## Ranking table

Run:

```sh
python3 05_ranktable.py
```