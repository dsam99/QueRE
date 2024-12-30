import torch
import numpy as np
from src.quere import ClosedEndedExplanationDataset, OpenEndedExplanationDataset, SquadExplanationDataset
from baselines.rep_dataset import RepDataset
from src.utils import get_mlp_results

import sys

import argparse

if __name__ == "__main__":

    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama-7b")
    parser.add_argument("--dataset", type=str, default="WinoGrande")
    parser.add_argument("--inv_cdf_norm", action="store_true", default=False, help="Use inverse cdf normalization")
    parser.add_argument("--random", action="store_true", default=False, help="Use random prompts")
    parser.add_argument("--gpt_exp", action="store_true", default=False, help="Use GPT explanations")
    parser.add_argument("--gpt_state", action="store_true", default=False, help="Use GPT state prompts")
    args = parser.parse_args()

    if args.dataset == "BooIQ":
        dataset = ClosedEndedExplanationDataset("BooIQ", args.llm, load_quere=True)
    elif args.dataset == "HaluEval":
        dataset = ClosedEndedExplanationDataset("HaluEval", args.llm, load_quere=True)
    elif args.dataset == "ToxicEval":
        dataset = ClosedEndedExplanationDataset("ToxicEval", args.llm, load_quere=True)    
    elif args.dataset == "CommonsenseQA":
        dataset = ClosedEndedExplanationDataset("CommonsenseQA", args.llm, load_quere=False)
    elif args.dataset == "WinoGrande":
        dataset = ClosedEndedExplanationDataset("WinoGrande", args.llm, load_quere=True)
    elif args.dataset == "nq":
        dataset = OpenEndedExplanationDataset(args.llm, load_quere=True)
    elif args.dataset == "squad":
        dataset = SquadExplanationDataset(args.llm, load_quere=True)
    
    b = True

    rep_dataset = RepDataset(args.dataset, args.llm)
    train_rep = rep_dataset.train_rep
    test_rep = rep_dataset.test_rep

    train_data, train_labels, train_log_probs = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs
    
    test_data, test_labels, test_log_probs, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs
    train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
    test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

    train_pre_conf = train_pre_conf.reshape(-1, 1)
    test_pre_conf = test_pre_conf.reshape(-1, 1)

    if "70b" in args.llm:
        # make all train only len 1000
        train_data = train_data[:1000]
        train_labels = train_labels[:1000]
        train_log_probs = train_log_probs[:1000]
        train_logits = train_logits[:1000]
        train_pre_conf = train_pre_conf[:1000]
        train_post_conf = train_post_conf[:1000]
        train_rep = train_rep[:1000]
    
    # reshape log probs
    train_log_probs = train_log_probs.reshape(len(train_data), -1)
    test_log_probs = test_log_probs.reshape(len(test_data), -1)


    #print shapes
    print(train_data.shape, train_labels.shape)
    print(train_log_probs.shape)

    results = {
        "logits_acc": [],
        "rep_acc": [],
        "logprob_acc": [],
        "preconf_acc": [],
        "postconf_acc": [],
        "exp_acc": [],
        "exp_all_acc": [],
        "logprob_f1": [],
        "rep_f1": [],
        "logits_f1": [],
        "preconf_f1": [],
        "postconf_f1": [],
        "exp_f1": [],
        "exp_all_f1": [],
        "logprob_ece": [],
        "rep_ece": [],
        "logits_ece": [],
        "preconf_ece": [],
        "postconf_ece": [],
        "exp_ece": [],
        "exp_all_ece": [],
        "logits_auroc": [],
        "rep_auroc": [],
        "logprob_auroc": [],
        "preconf_auroc": [],
        "postconf_auroc": [],
        "exp_auroc": [],
        "exp_all_auroc": [],
    }

    seeds = range(1)
    
    for seed in seeds:
    
        # set random seed   
        np.random.seed(seed)
        torch.manual_seed(seed)

        # get results for logprob
        acc, f1, ece, auroc = get_mlp_results(train_log_probs, train_labels, test_log_probs, test_labels, seed=seed, balanced=b)
        results["logprob_acc"].append(acc)
        results["logprob_f1"].append(f1)
        results["logprob_ece"].append(ece)
        results["logprob_auroc"].append(auroc)

        # get results for preconf
        acc, f1, ece, auroc = get_mlp_results(train_pre_conf, train_labels, test_pre_conf, test_labels, seed=seed, balanced=b)
        results["preconf_acc"].append(acc)
        results["preconf_f1"].append(f1)
        results["preconf_ece"].append(ece)
        results["preconf_auroc"].append(auroc)

        # get results for postconf
        acc, f1, ece, auroc = get_mlp_results(train_post_conf, train_labels, test_post_conf, test_labels, seed=seed, balanced=b)
        results["postconf_acc"].append(acc)
        results["postconf_f1"].append(f1)
        results["postconf_ece"].append(ece)
        results["postconf_auroc"].append(auroc)

        # get results for logits
        acc, f1, ece, auroc = get_mlp_results(train_logits, train_labels, test_logits, test_labels, seed=seed, balanced=b)
        results["logits_acc"].append(acc)
        results["logits_f1"].append(f1)
        results["logits_ece"].append(ece)
        results["logits_auroc"].append(auroc)

        # get results for exp
        acc, f1, ece, auroc = get_mlp_results(train_data, train_labels, test_data, test_labels, seed=seed, balanced=b)
        results["exp_acc"].append(acc)
        results["exp_f1"].append(f1)
        results["exp_ece"].append(ece)
        results["exp_auroc"].append(auroc)

        # get reuslts for exp_all
        train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
        test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

        acc, f1, ece, auroc = get_mlp_results(train_data_all, train_labels, test_data_all, test_labels, seed=seed, balanced=b)
        results["exp_all_acc"].append(acc)
        results["exp_all_f1"].append(f1)
        results["exp_all_ece"].append(ece)
        results["exp_all_auroc"].append(auroc)

        # get results for rep
        acc, f1, ece, auroc = get_mlp_results(train_rep, train_labels, test_rep, test_labels, seed=seed, balanced=b)
        results["rep_acc"].append(acc)
        results["rep_f1"].append(f1)
        results["rep_ece"].append(ece)
        results["rep_auroc"].append(auroc)

    # compute means
    results = {k: np.mean(v) for k, v in results.items()}
    results = {k: round(v, 4) for k, v in results.items()}
    # for k in ["logits_f1", "rep_f1", "logprob_f1", "preconf_f1", "postconf_f1", "exp_f1", "exp_all_f1"]:
    for k in ["logits_auroc", "rep_auroc", "preconf_auroc", "postconf_auroc", "logprob_auroc", "exp_all_auroc"]:
        print(k, results[k])    
