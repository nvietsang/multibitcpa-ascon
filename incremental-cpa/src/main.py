from focpa import focpa_v6
# from socpa import socpa_v9
import numpy as np
import argparse
import random
import os

def main():
    parser = argparse.ArgumentParser()

    # Task
    TASK_FOCPA = 0
    TASK_SOCPA = 1
    parser.add_argument("--task",
                        dest="task",
                        help="focpa (0) | socpa (1)",
                        type=int,
                        required=True)    
    parser.add_argument("--debug",
                        dest="debug",
                        help="Print more information in debug mode",
                        action="store_true")
    parser.add_argument('--selection-function', 
                        dest='selection_function', 
                        help="[y] for Sbox output, [z] for fine-tuned linear layer output",
                        type=str,
                        default="z")

    # Key guess
    parser.add_argument("--n-bits-selection-function",
                        dest="nd",
                        help="Number of bits for selection function",
                        type=int)
    parser.add_argument("--n-key-guesses",
                        dest="nk",
                        help="Number of key guesses. Calculated automatically nk=1<<(nd*3) if not specified.",
                        type=int)
    parser.add_argument("--intersection-recovery",
                        dest="use_intersection_recovery",
                        help="Use intersection recovery for nd > 1",
                        action="store_true")
    parser.add_argument('--target-key', 
                        dest='target_key', 
                        help="Target first half or second half of key [k0 | k1]?",
                        type=str,
                        default="k0")
    parser.add_argument('--first-subkey-index', 
                        dest='i0', 
                        help="First index of subkey bit",
                        nargs="+")
    parser.add_argument('--n-ranks', 
                        dest='nr', 
                        help="Number of candidates shown for ranking",
                        type=int,
                        default=5)

    # Traces
    parser.add_argument("--n-traces",
                        dest="nn",
                        help="Number of traces",
                        type=int,
                        required=True)
    parser.add_argument("--to-shuffle",
                        dest="to_shuffle",
                        help="Shuffle the traces? (for success rate measurement)",
                        action="store_true")
    parser.add_argument("--space",
                        dest="space",
                        help="Space of shuffling (for success rate measurement)",
                        type=int,
                        default=200000)
    parser.add_argument("--n-repetition",
                        dest="nrep",
                        help="Number of repetitions (for success rate measurement)",
                        type=int,
                        default=1)
    parser.add_argument("--start-sample",
                        dest="nt_s",
                        help="Start sample index for trimming traces",
                        type=int)
    parser.add_argument("--end-sample",
                        dest="nt_e",
                        help="End sample index for trimming traces",
                        type=int)
    parser.add_argument("--n-samples",
                        dest="nt",
                        help="Number of samples in each trace. Calculated automatically nt=nt_e-nt_s if not specified",
                        type=int)
    parser.add_argument("--step",
                        dest="ns",
                        help="Number of traces for each process",
                        type=int)
    parser.add_argument("--window",
                        dest="nw",
                        help="Window size for second-order CPA",
                        type=int)
    parser.add_argument('--path-to-traces', 
                        dest='path_to_traces', 
                        help="Path to the folder containing traces",
                        type=str)
    parser.add_argument('--data-type', 
                        dest='data_type', 
                        help="data type for samples: float32 or float64",
                        type=str)

    # Files
    parser.add_argument('--path-to-nonces', 
                        dest='path_to_nonces', 
                        help="Path to the single numpy file of nonces",
                        type=str)
    parser.add_argument('--path-to-checkpoints', 
                        dest='path_to_checkpoints', 
                        help="Path to the folder of checkpoints",
                        type=str)
    


    config = parser.parse_args()
    
    # Verify the parameter nk (number of key guesses).
    # It depends on bitsize of intermediate variables.
    if not config.nk: 
        if   config.selection_function in ["z0", "z1", "z4"]: config.nk = 1 << (config.nd * 3)
        elif config.selection_function in ["y0", "y1", "y4"]: config.nk = 1 << (config.nd * 2)
        else: raise ValueError("Undefined selection function")
    else: 
        if   config.selection_function in ["z0", "z1", "z4"]: assert config.nk == 1 << (config.nd * 3)
        elif config.selection_function in ["y0", "y1", "y4"]: assert config.nk == 1 << (config.nd * 2)
        else: raise ValueError("Undefined selection function")
        
    # Verify the parameter nt (number of samples considered in each trace)
    #   nt_e: ending sample index
    #   nt_s: starting sample index
    # nt_s and nt_e are determined in advance by some observations 
    if not config.nt: config.nt = config.nt_e - config.nt_s
    else: assert config.nt == config.nt_e - config.nt_s

    # Choose the appropriate data type to save memory, 
    # and to ensure the precision level in computation
    if config.data_type == "float32": config.data_type = np.float32
    elif config.data_type == "float64": config.data_type = np.float64
    else: raise ValueError("Undefined data type")

    path_prefix = f"{config.path_to_checkpoints}/{config.target_key}/{config.selection_function}/b{config.nd}"
    key_indexes = [int(v) for v in config.i0]
    
    if config.task == TASK_FOCPA:
        for rep in range(config.nrep):
            # Randomize indexes of traces
            assert config.nn <= config.space
            if config.to_shuffle:
                trace_indexes = random.sample(range(config.space), config.nn)
            else:
                trace_indexes = [_ for _ in range(config.nn)]
            for i0 in key_indexes:
                config.i0 = i0
                if not os.path.exists(f"{path_prefix}/{i0:02d}/{rep:02d}"): os.makedirs(f"{path_prefix}/{i0:02d}/{rep:02d}")
                print(f"[INFO] Index {i0:02d}. Repetition {rep:02d}/{config.nrep:02d}")
                config.path_to_checkpoints = f"{path_prefix}/{i0:02d}/{rep:02d}"
                focpa_v6(config, trace_indexes)
    
    # elif config.task == TASK_SOCPA:
    #     for rep in range(config.nrep):
    #         # Randomize indexes of traces
    #         assert config.nn <= config.space
    #         trace_indexes = random.sample(range(config.space), config.nn)
    #         for i0 in key_indexes:
    #             config.i0 = i0
    #             if not os.path.exists(f"{path_prefix}/{i0:02d}"): os.makedirs(f"{path_prefix}/{i0:02d}")
    #             print(f"[INFO] Index {i0:02d}. Repetition {rep:02d}/{config.nrep}")
    #             config.path_to_checkpoints = f"{path_prefix}/{i0:02d}/{rep:02d}"
    #             socpa_v9(config, trace_indexes)

    else: raise ValueError("Undefined task")

    

if __name__ == "__main__":
    main()