import argparse
import numpy as np
import torch
from baselines.rep_dataset import RepDataset

if __name__ == "__main__":

    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama3-3b")
    parser.add_argument("--dataset", type=str, default="WinoGrande")
    args = parser.parse_args()
    
    rep_dataset = RepDataset(args.dataset, args.llm)