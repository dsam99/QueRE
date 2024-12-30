import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from scipy.stats import norm

from src.quere import OpenEndedExplanationDataset, SquadExplanationDataset
from baselines.rep_dataset import RepDataset
from src.utils import train_linear_model, compute_ece, normalize_data, get_linear_results
from src.llm import load_llm

import sys

import argparse

if __name__ == "__main__":

    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama-7b")
    parser.add_argument("--inv_cdf_norm", action="store_true", default=False, help="Use inverse cdf normalization")
    parser.add_argument("--balance", action="store_true", default=False, help="Balance data")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset to use")
    parser.add_argument("--gpt_exp", action="store_true", default=False, help="Use GPT explanations")
    parser.add_argument("--gpt_state", action="store_true", default=False, help="Use GPT state prompts")
    parser.add_argument("--random", action="store_true", default=False, help="Use random")
    
    args = parser.parse_args()
    b = True

    if args.dataset == "nq":
        dataset = OpenEndedExplanationDataset(args.llm, load_quere=True)
    elif args.dataset == "squad":
        dataset = SquadExplanationDataset(args.llm, load_quere=True)
    
    rep_dataset = RepDataset(args.dataset, args.llm)
    train_rep = rep_dataset.train_rep
    test_rep = rep_dataset.test_rep

    train_data, train_labels, train_log_probs = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs
    
    test_data, test_labels, test_log_probs, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs
    train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
    test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs


    # print label means
    print(train_labels.mean(), test_labels.mean())

    results = {
        "logits_acc": [],
        "rep_acc": [],
        "logprob_acc": [],
        "preconf_acc": [],
        "postconf_acc": [],
        "exp_acc": [],
        "exp_all_acc": [],
        "logits_f1": [],
        "rep_f1": [],
        "logprob_f1": [],
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

    seeds = range(5)

    # unsqueeze 2nd dim of 1d outputs
    train_pre_conf = train_pre_conf.reshape(-1, 1)
    test_pre_conf = test_pre_conf.reshape(-1, 1)
    train_post_conf = train_post_conf.reshape(-1, 1)
    test_post_conf = test_post_conf.reshape(-1, 1)
    train_log_probs = train_log_probs.reshape(-1, 1)
    test_log_probs = test_log_probs.reshape(-1, 1)

    # standard z-score normalize all data with train mean and std
    train_data, test_data = normalize_data(train_data, test_data)

    print(min(train_log_probs))
    # clip log probs
    train_log_probs = np.clip(train_log_probs, -10000, 0)
    test_log_probs = np.clip(test_log_probs, -10000, 0)

    train_log_probs, test_log_probs = normalize_data(train_log_probs, test_log_probs)
    train_pre_conf, test_pre_conf = normalize_data(train_pre_conf, test_pre_conf)
    train_post_conf, test_post_conf = normalize_data(train_post_conf, test_post_conf)
    train_logits, test_logits = normalize_data(train_logits, test_logits)
    train_rep, test_rep = normalize_data(train_rep, test_rep)

    for seed in seeds:
    
        # set random seed   
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # get results for logprob
        acc, f1, ece, auroc = get_linear_results(train_log_probs, train_labels, test_log_probs, test_labels, seed=seed, balanced=b)
        results["logprob_acc"].append(acc)
        results["logprob_f1"].append(f1)
        results["logprob_ece"].append(ece)
        results["logprob_auroc"].append(auroc)

        # get results for preconf
        acc, f1, ece, auroc = get_linear_results(train_pre_conf, train_labels, test_pre_conf, test_labels, seed=seed, balanced=b)
        results["preconf_acc"].append(acc)
        results["preconf_f1"].append(f1)
        results["preconf_ece"].append(ece)
        results["preconf_auroc"].append(auroc)

        # get results for postconf
        acc, f1, ece, auroc = get_linear_results(train_post_conf, train_labels, test_post_conf, test_labels, seed=seed, balanced=b)
        results["postconf_acc"].append(acc)
        results["postconf_f1"].append(f1)
        results["postconf_ece"].append(ece)
        results["postconf_auroc"].append(auroc)

        # get results for logits
        acc, f1, ece, auroc = get_linear_results(train_logits, train_labels, test_logits, test_labels, seed=seed, balanced=b)
        results["logits_acc"].append(acc)
        results["logits_f1"].append(f1)
        results["logits_ece"].append(ece)
        results["logits_auroc"].append(auroc)

        # get results for exp
        clf = train_linear_model(train_data, train_labels, test_data, test_labels, seed=seed, balanced=b)
        results["exp_acc"].append(acc)
        results["exp_f1"].append(f1)
        results["exp_ece"].append(ece)
        results["exp_auroc"].append(auroc)

        # get reuslts for exp_all
        train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
        test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

        acc, f1, ece, auroc = get_linear_results(train_data_all, train_labels, test_data_all, test_labels, seed=seed, balanced=b)
        results["exp_all_acc"].append(acc)
        results["exp_all_f1"].append(f1)
        results["exp_all_ece"].append(ece)
        results["exp_all_auroc"].append(auroc)

        # get results for rep
        acc, f1, ece, auroc = get_linear_results(train_rep, train_labels, test_rep, test_labels, seed=seed, balanced=b)
        results["rep_acc"].append(acc)
        results["rep_f1"].append(f1)
        results["rep_ece"].append(ece)
        results["rep_auroc"].append(auroc)

    # compute means
    results = {k: np.mean(v) for k, v in results.items()}
    results = {k: round(v, 4) for k, v in results.items()}
    for k in ["logits_auroc", "rep_auroc", "preconf_auroc", "postconf_auroc", "logprob_auroc", "exp_all_auroc"]:
        print(k, results[k])    
