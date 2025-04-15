# Incremental CPA

This folder contains the source code of the first-order CPA.

## Scripts

There are 6 scripts in this folder. Each corresponds to a configuration for the first-order CPA (focpa).

- `focpa_k0b1z4`
- `focpa_k0b2z4`
- `focpa_k0b3z4`
- `focpa_k1b1z4`
- `focpa_k1b2z4`
- `focpa_k1b3z4`

For example, `focpa_k0b1z4` means the CPA aims to recover the first half $k_0$ of the key, using 1-bit selection function and $z_4$ as the attack point.

## Configuration

An example of parameters for the CPA:

```
time python3 main.py --task 0\
                     --selection-function z4\
                     --n-bits-selection-function 1\
                     --target-key k0\
                     --first-subkey-index 1 6 7 9 11 12 17 22 23 25 27 28 33 34 38 39 43 44 48 49 54 55 59 60\
                     --n-rank 8\
                     --n-traces 10000\
                     --n-repetition 3\
                     --space 10000\
                     --path-to-traces ../bi32-armv6/traces\
                     --start-sample 150\
                     --end-sample 500\
                     --step 200\
                     --data-type float64\
                     --path-to-nonces ../bi32-armv6/nonces.npy\
                     --path-to-checkpoints ../bi32-armv6/checkpoints\
                    #   --to-shuffle
```