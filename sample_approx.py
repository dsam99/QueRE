import torch
import numpy as np
from sklearn.metrics import f1_score
import sys
from scipy.stats import norm

from data.openai_dataset import SquadExplanationDataset_OAI, MCQExplanationDataset_OAI, BooIQExplanationDataset_OAI, WinoGrandeExplanationDataset_OAI, OpenEndedExplanationDataset_OAI
from data.rep_dataset import RepDataset
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
    parser.add_argument("--gpt_state", action="store_true", default=False, help="Use GPT state prompts")
    parser.add_argument("--adv", action="store_true", default=False, help="Use adv system prompt")
    args = parser.parse_args()
    b = True # always balance!

    llm = "gpt-3.5-turbo-0125"

    if args.dataset == "nq":
        dataset = OpenEndedExplanationDataset_OAI(llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state)
        append_dataset = OpenEndedExplanationDataset_OAI(llm, gpt_exp=True)
    elif args.dataset == "BooIQ":
        dataset = BooIQExplanationDataset_OAI("BooIQ", llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state, adv=args.adv)
        append_dataset = BooIQExplanationDataset_OAI("BooIQ", llm, gpt_exp=True)
    elif args.dataset == "squad":
        dataset = SquadExplanationDataset_OAI(llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state)
        append_dataset = SquadExplanationDataset_OAI(llm, gpt_exp=True)
    elif args.dataset == "cs_qa":
        dataset = MCQExplanationDataset_OAI("CommonsenseQA", llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state)
        append_dataset = MCQExplanationDataset_OAI("CommonsenseQA", llm, gpt_exp=True)
    elif args.dataset == "ToxicEval":
        dataset = BooIQExplanationDataset_OAI("ToxicEval", llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state, adv=args.adv)
        append_dataset = BooIQExplanationDataset_OAI("ToxicEval", llm, gpt_exp=True)
    elif args.dataset == "HaluEval":
        dataset = BooIQExplanationDataset_OAI("HaluEval", llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state, adv=args.adv)
        append_dataset = BooIQExplanationDataset_OAI("HaluEval", llm, gpt_exp=True)
    elif args.dataset == "WinoGrande":
        dataset = WinoGrandeExplanationDataset_OAI("WinoGrande", llm, gpt_exp=args.gpt_exp, gpt_state=args.gpt_state)
        append_dataset = WinoGrandeExplanationDataset_OAI("WinoGrande", llm, gpt_exp=True)
    else:
        print(args.dataset + " not recognized")
    
    train_data, train_labels, train_log_probs = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs    
    test_data, test_labels, test_log_probs, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs
    
    train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
    test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

    # append gpt explanations
    train_data = np.concatenate([train_data, append_dataset.train_data], axis=1)
    test_data = np.concatenate([test_data, append_dataset.test_data], axis=1)

    # print label means
    print("label means", train_labels.mean(), test_labels.mean())

    seeds = range(5)

    # unsqueeze 2nd dim of 1d outputs
    train_pre_conf = train_pre_conf.reshape(train_labels.shape[0], -1)
    test_pre_conf = test_pre_conf.reshape(test_labels.shape[0], -1)
    train_post_conf = train_post_conf.reshape(train_labels.shape[0], -1)
    test_post_conf = test_post_conf.reshape(test_labels.shape[0], -1)
    train_log_probs = train_log_probs.reshape(train_labels.shape[0], -1)
    test_log_probs = test_log_probs.reshape(test_labels.shape[0], -1)


    ks = range(5, 30, 5)

    gts = np.zeros(len(ks))
    approxs = np.zeros(len(ks))

    gts_std = np.zeros(len(ks))
    approxs_std = np.zeros(len(ks))

    for k_ind, k in enumerate(ks):


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
            "approx_all_auroc": [],
         }

        for seed in seeds:
    
            # set random seed   
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # construct approximation
            train_approx = np.zeros_like(train_data)
            test_approx = np.zeros_like(test_data)

            for i in range(train_data.shape[1]):
                train_approx[:, i] = np.random.binomial(p=train_data[:, i], n=k) / k

            for i in range(test_data.shape[1]):
                test_approx[:, i] = np.random.binomial(p=test_data[:, i], n=k) / k

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
            train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf, train_logits], axis=1)
            test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf, test_logits], axis=1)

            train_approx_all = np.concatenate([train_approx, train_log_probs, train_pre_conf, train_post_conf, train_logits], axis=1)
            test_approx_all = np.concatenate([test_approx, test_log_probs, test_pre_conf, test_post_conf, test_logits], axis=1)

            acc, f1, ece, auroc = get_linear_results(train_data_all, train_labels, test_data_all, test_labels, seed=seed, balanced=b)
            results["exp_all_acc"].append(acc)
            results["exp_all_f1"].append(f1)
            results["exp_all_ece"].append(ece)
            results["exp_all_auroc"].append(auroc)

            acc, f1, ece, auroc = get_linear_results(train_approx_all, train_labels, test_approx_all, test_labels, seed=seed, balanced=b)
            results["approx_all_auroc"].append(auroc)

        # compute means
        mean_results = {k: np.mean(v) for k, v in results.items()}
        mean_results = {k: np.round(v, 4) for k, v in mean_results.items()}
        for name in ["approx_all_auroc", "exp_all_auroc"]:
            print(name, mean_results[name])    
        
        gts[k_ind] = mean_results["exp_all_auroc"]
        approxs[k_ind] = mean_results["approx_all_auroc"]

        # compute stds
        std_results = {k: np.std(v) / np.sqrt(len(seeds)) for k, v in results.items()}
        std_results = {k: np.round(v, 4) for k, v in std_results.items()}

        gts_std[k_ind] = std_results["exp_all_auroc"]
        approxs_std[k_ind] = std_results["approx_all_auroc"]

    print("gts", gts)
    print("approx", approxs)

    colors = [(0.578, 0.747, 0.802), (0.758, 0.617, 0.849), (0.900, 0.613, 0.656)]

    # plot 
    import matplotlib.pyplot as plt
    plt.plot(ks, gts, label="LLM Probabilities", color=colors[0], linewidth=2)
    plt.fill_between(ks, gts - gts_std, gts + gts_std, alpha=0.2, color=colors[0])
    plt.plot(ks, approxs, label="Sampling", color=colors[1], linewidth=2)
    plt.fill_between(ks, approxs - approxs_std, approxs + approxs_std, alpha=0.2, color=colors[1])



    # increase font size
    plt.xticks(ks, fontsize=14)
    plt.yticks(fontsize=14)
    
    if args.dataset == "HaluEval":
        plt.legend(fontsize=16)

    plt.xlabel("Number of Samples", fontsize=24)
    plt.ylabel("AUROC", fontsize=24)

    # fix plot getting cut off
    plt.tight_layout()

    plt.savefig("figs/sample_approx_" + args.dataset + ".png")
    plt.savefig("figs/sample_approx_" + args.dataset + ".pdf")
