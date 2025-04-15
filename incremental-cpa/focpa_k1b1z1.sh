time python3 main.py --task 0\
                      --selection-function z1\
                      --n-bits-selection-function 1\
                      --target-key k1\
                      --first-subkey-index 3 5 6 7 13 15 17 22 24 26 32 33 34 39 41 43 45 50 51 52 53 58 60 62\
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