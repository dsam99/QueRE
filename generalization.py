import torch
import numpy as np
from src.quere import ClosedEndedExplanationDataset, SquadExplanationDataset, OpenEndedExplanationDataset
from baselines.rep_dataset import RepDataset
from src.utils import get_linear_acc_weights
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
        dataset = ClosedEndedExplanationDataset("CommonsenseQA", args.llm, load_quere=True)
    elif args.dataset == "WinoGrande":
        dataset = ClosedEndedExplanationDataset("WinoGrande", args.llm, load_quere=True)
    elif args.dataset == "squad":
        dataset = SquadExplanationDataset(args.llm, load_quere=True)
    elif args.dataset == "nq":
        dataset = OpenEndedExplanationDataset(args.llm, load_quere=True)

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

    train_pre_conf = train_pre_conf.reshape(len(train_labels), -1)
    test_pre_conf = test_pre_conf.reshape(len(test_labels), -1)

    train_log_probs = train_log_probs.reshape(len(train_labels), -1)
    test_log_probs = test_log_probs.reshape(len(test_labels), -1)




    # print(train_labels.mean(), test_labels.mean())
    print(train_data[:2])

    #print shapes
    print(train_data.shape, train_labels.shape)
    results = {}

    for k in ["logprob", "logits", "exp_all", "rep"]:
        results[k + "_acc"] = []
        results[k + "_bound"] = []

    seeds = range(5)
    # get reuslts for exp_all
    train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
    test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

    for seed in seeds:
    
        # set random seed   
        np.random.seed(seed)
        torch.manual_seed(seed)

        # balance data to have equal number of positive and negative examples
        pos_idx = np.where(train_labels == 1)[0]
        neg_idx = np.where(train_labels == 0)[0]
        
        num_min = min(len(pos_idx), len(neg_idx))
        pos_idx = np.random.choice(pos_idx, num_min, replace=False)
        neg_idx = np.random.choice(neg_idx, num_min, replace=False)

        test_num_min = min(np.sum(test_labels), len(test_labels) - np.sum(test_labels))
        pos_idx_test = np.where(test_labels == 1)[0]
        neg_idx_test = np.where(test_labels == 0)[0]
        pos_idx_test = np.random.choice(pos_idx_test, test_num_min, replace=False)
        neg_idx_test = np.random.choice(neg_idx_test, test_num_min, replace=False)

        test_idx = np.concatenate([pos_idx_test, neg_idx_test])
        test_data_all_b, test_labels_b, test_log_probs_b, test_rep_b = test_data_all[test_idx], test_labels[test_idx], test_log_probs[test_idx], test_rep[test_idx]
        test_logits_b, test_pre_conf_b, test_post_conf_b = test_logits[test_idx], test_pre_conf[test_idx], test_post_conf[test_idx]
        
        idx = np.concatenate([pos_idx, neg_idx])
        train_data_all_b, train_labels_b, train_log_probs_b, train_rep_b = train_data_all[idx], train_labels[idx], train_log_probs[idx], train_rep[idx]
        train_logits_b, train_pre_conf_b, train_post_conf_b = train_logits[idx], train_pre_conf[idx], train_post_conf[idx]

        N = train_data_all_b.shape[0]
        # N = 200
        delta = 0.01
        variances = np.arange(0.1, 10, 0.01)

        data_names = [
            ("logprob", train_log_probs_b, test_log_probs_b),
            ("logits", train_logits_b, test_logits_b),
            ("rep", train_rep_b, test_rep_b),
            ("exp_all", train_data_all_b, test_data_all_b),
        ]

        print("shapes", train_log_probs_b.shape, train_logits_b.shape, train_rep_b.shape, train_data_all_b.shape)
        for name, train_d, test_d in data_names:

            # get results for logprob
            acc, og_weights, og_bias, final_weights, final_bias = get_linear_acc_weights(train_d, train_labels_b, test_d, test_labels_b, seed=seed)
            results[name + "_acc"].append(acc)
            
            # compute bound term 
            bounds = []
            weight_diff = np.square(og_bias - final_bias) + np.sum(np.square(og_weights - final_weights))
            for v in variances:
                b = np.sqrt( (weight_diff / (4 * (v**2)) + np.log(N / delta) + 10) / (N - 1)    )
                bounds.append(b)
            bound_term = np.min(bounds)
            results[name + "_bound"].append(bound_term)

    # compute means
    results = {k: np.mean(v) for k, v in results.items()}
    results = {k: round(v, 4) for k, v in results.items()}

    for k in ["logits", "rep", "logprob", "exp_all"]:
        print(k, max(results[k + "_acc"] - results[k + "_bound"], 0))
