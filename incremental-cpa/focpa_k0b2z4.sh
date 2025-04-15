time python3 main.py --task 0\
                      --selection-function z4\
                      --n-bits-selection-function 2\
                      --target-key k0\
                      --first-subkey-index 3 9 14 19 24 30 35 41 46 51 56 62\
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