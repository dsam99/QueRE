import torch
import numpy as np
from baselines.semantic_dataset import BoolSemanticDataset, WinoGrandeSemanticDataset, MCQSemanticDataset
from src.utils import get_linear_results
from src.llm import load_llm

import sys

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama3-8b")
    parser.add_argument("--dataset", type=str, default="BooIQ")
    args = parser.parse_args()
    
    if args.dataset == "WinoGrande" or args.dataset == "CommonsenseQA":
        
        if args.dataset == "WinoGrande":
            train_dataset = WinoGrandeSemanticDataset(args.llm, split="train")
            test_dataset = WinoGrandeSemanticDataset(args.llm, split="test")

            train_labels, train_post_conf = \
                train_dataset.labels, train_dataset.post_confs

            test_labels, test_post_conf, = \
                test_dataset.labels, test_dataset.post_confs

        elif args.dataset == "CommonsenseQA":
            train_dataset = MCQSemanticDataset("CommonsenseQA", args.llm, split="train")
            test_dataset = MCQSemanticDataset("CommonsenseQA", args.llm, split="test")

            train_labels, train_post_conf = \
                train_dataset.labels, train_dataset.option_probs
            test_labels, test_post_conf, = \
                test_dataset.labels, test_dataset.option_probs

    else:
        if args.dataset == "BooIQ":
            dataset = BoolSemanticDataset("BooIQ", args.llm)
        elif args.dataset == "HaluEval":
            dataset = BoolSemanticDataset("HaluEval", args.llm)
        elif args.dataset == "ToxicEval":
            dataset = BoolSemanticDataset("ToxicEval", args.llm)

        train_labels, train_post_conf = \
            dataset.train_labels, dataset.train_post_confs
        
        test_labels, test_post_conf, = \
            dataset.test_labels, dataset.test_post_confs
    
    # get results for postconf
    acc, f1, ece, auroc = get_linear_results(train_post_conf, train_labels, test_post_conf, test_labels, seed=0, balanced=True)
    print("Semantic Sim Auroc:", auroc)