import torch
import numpy as np
from src.quere import ClosedEndedExplanationDataset
from src.utils import get_linear_results

import argparse

if __name__ == "__main__":

    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama3-3b")
    parser.add_argument("--dataset", type=str, default="WinoGrande")
    args = parser.parse_args()

    if args.dataset == "BooIQ":
        dataset = ClosedEndedExplanationDataset("BooIQ", args.llm, load_quere=True)
    elif args.dataset == "HaluEval":
        dataset = ClosedEndedExplanationDataset("HaluEval", args.llm, load_quere=True)
    elif args.dataset == "ToxicEval":
        dataset = ClosedEndedExplanationDataset("ToxicEval", args.llm, load_quere=True)    
    elif args.dataset == "CommonsenseQA":
        dataset = ClosedEndedExplanationDataset("CommonsenseQA", args.llm, load_quere=True)
    elif args.dataset == "winogrande":
        dataset = ClosedEndedExplanationDataset("WinoGrande", args.llm, load_quere=True)

    b = True

    train_data, train_labels, train_log_probs = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs
    
    test_data, test_labels, test_log_probs, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs

    train_pre_conf = dataset.train_pre_confs.reshape(-1, 1)
    test_pre_conf = dataset.test_pre_confs.reshape(-1, 1)

    train_post_conf = dataset.train_post_confs.reshape(-1, 1)
    test_post_conf = dataset.test_post_confs.reshape(-1, 1)

    test_results = {
        "postconf_auroc": [],
        "topk_auroc": [],
        "cot_auroc": [], 
        "multistep_auroc": [],
        "concat_baseline_auroc": [],
        "quere_auroc": [],
        "quere_all_auroc": []
    }

    if args.dataset in ["winogrande", "BooIQ"]:
        test_prefix = args.dataset + "_validation.npy"
    else:
        test_prefix = "halueval_test.npy"

    train_cot = np.load("/home/dylansam/repos/llm-uncertainty/cot_" + args.llm + "/" + args.dataset + "_train.npy")
    test_cot = np.load("/home/dylansam/repos/llm-uncertainty/cot_" + args.llm + "/" + test_prefix)

    train_topk = np.load("/home/dylansam/repos/llm-uncertainty/top_k_" + args.llm + "/"  + args.dataset + "_train.npy")
    test_topk = np.load("/home/dylansam/repos/llm-uncertainty/top_k_" + args.llm + "/"  +  test_prefix)

    train_multistep = np.load("/home/dylansam/repos/llm-uncertainty/multistep_" + args.llm + "/"  + args.dataset + "_train.npy")
    test_multistep = np.load("/home/dylansam/repos/llm-uncertainty/multistep_" + args.llm + "/"  + test_prefix)

    # truncate all to length of train_cot
    train_data = train_data[:train_cot.shape[0]]
    train_labels = train_labels[:train_cot.shape[0]]
    train_log_probs = train_log_probs[:train_cot.shape[0]]
    train_pre_conf = train_pre_conf[:train_cot.shape[0]]
    train_post_conf = train_post_conf[:train_cot.shape[0]]

    # truncate length to max of both
    test_shape_og = test_data.shape[0]
    test_shape_new = test_cot.shape[0]

    min_shape = min(test_shape_og, test_shape_new)

    test_data = test_data[:min_shape]
    test_labels = test_labels[:min_shape]
    test_log_probs = test_log_probs[:min_shape]
    test_pre_conf = test_pre_conf[:min_shape]
    test_post_conf = test_post_conf[:min_shape]
    test_cot = test_cot[:min_shape]
    test_topk = test_topk[:min_shape]
    test_multistep = test_multistep[:min_shape]


    print("train")
    print(train_data.shape, train_labels.shape, train_log_probs.shape, train_pre_conf.shape, train_post_conf.shape)
    print(train_cot.shape, train_topk.shape, train_multistep.shape)

    print("test")
    print(test_data.shape, test_labels.shape, test_log_probs.shape, test_pre_conf.shape, test_post_conf.shape)
    print(test_cot.shape, test_topk.shape, test_multistep.shape)

    seeds = range(1)
    
    for seed in seeds:
    
        # set random seed   
        np.random.seed(seed)
        torch.manual_seed(seed)

        # get results for logits
        acc, f1, ece, auroc = get_linear_results(train_post_conf, train_labels, test_post_conf, test_labels, seed=seed, balanced=b, C=1)
        test_results["postconf_auroc"].append(auroc)

        # get reuslts for quere
        train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
        test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

        acc, f1, ece, auroc = get_linear_results(train_data_all, train_labels, test_data_all, test_labels, seed=seed, balanced=b, C=1)
        test_results["quere_auroc"].append(auroc)

        # get results for 
        acc, f1, ece, auroc = get_linear_results(train_cot, train_labels, test_cot, test_labels, seed=seed, balanced=b, C=1)
        test_results["cot_auroc"].append(auroc)

        acc, f1, ece, auroc = get_linear_results(train_topk, train_labels, test_topk, test_labels, seed=seed, balanced=b, C=1)
        test_results["topk_auroc"].append(auroc)

        acc, f1, ece, auroc = get_linear_results(train_multistep, train_labels, test_multistep, test_labels, seed=seed, balanced=b, C=1)
        test_results["multistep_auroc"].append(auroc)

        baselines_train_all = np.concatenate([train_post_conf, train_cot, train_topk, train_multistep], axis=1)
        baselines_test_all = np.concatenate([test_post_conf, test_cot, test_topk, test_multistep], axis=1)

        acc, f1, ece, auroc = get_linear_results(baselines_train_all, train_labels, baselines_test_all, test_labels, seed=seed, balanced=b, C=1)
        test_results["concat_baseline_auroc"].append(auroc)

        quere_train_all = np.concatenate([train_data_all, train_cot, train_topk, train_multistep], axis=1)
        quere_test_all = np.concatenate([test_data_all, test_cot, test_topk, test_multistep], axis=1)

        acc, f1, ece, auroc = get_linear_results(quere_train_all, train_labels, quere_test_all, test_labels, seed=seed, balanced=b, C=1)
        test_results["quere_all_auroc"].append(auroc)

    for k in ["topk_auroc", "cot_auroc", "multistep_auroc", "concat_baseline_auroc"]:
        print(k, "Test", test_results[k])    
