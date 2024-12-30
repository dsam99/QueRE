import torch
import numpy as np
from src.quere_oai import SquadExplanationDataset_OAI, MCQExplanationDataset_OAI, BooIQExplanationDataset_OAI, WinoGrandeExplanationDataset_OAI, OpenEndedExplanationDataset_OAI
from src.utils import train_linear_model, compute_ece, normalize_data, get_linear_results

import argparse

if __name__ == "__main__": 
    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="gpt-3.5")
    parser.add_argument("--dataset", type=str, default="squad", help="Dataset to use")
    parser.add_argument("--gpt_exp", action="store_true", default=False, help="Use GPT explanations")
    args = parser.parse_args()
    b = True # always balance!

    llm = "gpt-3.5-turbo-0125" if args.llm == "gpt-3.5" else "gpt-4o-mini" # either gpt-3.5 or 4

    if args.dataset == "nq":
        dataset = OpenEndedExplanationDataset_OAI(llm, load_quere=True)
    elif args.dataset == "BooIQ":
        dataset = BooIQExplanationDataset_OAI("BooIQ", llm, load_quere=True)
    elif args.dataset == "squad":
        dataset = SquadExplanationDataset_OAI(llm, load_quere=True)
    elif args.dataset == "cs_qa":
        dataset = MCQExplanationDataset_OAI("CommonsenseQA", llm, load_quere=True)
    elif args.dataset == "ToxicEval":
        dataset = BooIQExplanationDataset_OAI("ToxicEval", llm, load_quere=True)
    elif args.dataset == "HaluEval":
        dataset = BooIQExplanationDataset_OAI("HaluEval", llm, load_quere=True)
    elif args.dataset == "WinoGrande":
        dataset = WinoGrandeExplanationDataset_OAI("WinoGrande", llm, load_quere=True)
    else:
        print(args.dataset + " not recognized")
    
    train_data, train_labels, train_log_probs = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs    
    test_data, test_labels, test_log_probs, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs
    
    train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
    test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

    train_sorted_logits, test_sorted_logits = dataset.train_sorted_logits, dataset.test_sorted_logits

    # print label means
    print("label means", train_labels.mean(), test_labels.mean())

    results = {
        "logprob_acc": [],
        "logits_acc": [],
        "preconf_acc": [],
        "postconf_acc": [],
        "exp_acc": [],
        "exp_all_acc": [],
        "logprob_f1": [],
        "logits_f1": [],
        "preconf_f1": [],
        "postconf_f1": [],
        "exp_f1": [],
        "exp_all_f1": [],
        "logprob_ece": [],
        "logits_ece": [],
        "preconf_ece": [],
        "postconf_ece": [],
        "exp_ece": [],
        "exp_all_ece": [],
        "logprob_auroc": [],
        "logits_auroc": [],
        "preconf_auroc": [],
        "postconf_auroc": [],
        "exp_auroc": [],
        "exp_all_auroc": [],
    }

    seeds = range(1)

    # unsqueeze 2nd dim of 1d outputs
    train_pre_conf = train_pre_conf.reshape(train_labels.shape[0], -1)
    test_pre_conf = test_pre_conf.reshape(test_labels.shape[0], -1)
    train_post_conf = train_post_conf.reshape(train_labels.shape[0], -1)
    test_post_conf = test_post_conf.reshape(test_labels.shape[0], -1)
    train_log_probs = train_log_probs.reshape(train_labels.shape[0], -1)
    test_log_probs = test_log_probs.reshape(test_labels.shape[0], -1)

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
        acc, f1, ece, auroc = get_linear_results(train_data, train_labels, test_data, test_labels, seed=seed, balanced=b)
        results["exp_acc"].append(acc)
        results["exp_f1"].append(f1)
        results["exp_ece"].append(ece)
        results["exp_auroc"].append(auroc)

        # get reuslts for exp_all
        train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf, train_sorted_logits], axis=1)
        test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf, test_sorted_logits], axis=1)

        acc, f1, ece, auroc = get_linear_results(train_data_all, train_labels, test_data_all, test_labels, seed=seed, balanced=b, C=1)
        results["exp_all_acc"].append(acc)
        results["exp_all_f1"].append(f1)
        results["exp_all_ece"].append(ece)
        results["exp_all_auroc"].append(auroc)


    # compute means
    results = {k: np.mean(v) for k, v in results.items()}
    results = {k: round(v, 4) for k, v in results.items()}
    for k in ["logits_auroc", "preconf_auroc", "postconf_auroc", "logprob_auroc", "exp_all_auroc"]:
        print(k, results[k])    
