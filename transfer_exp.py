import torch
import numpy as np
from src.quere import OpenEndedExplanationDataset, SquadExplanationDataset, ClosedEndedExplanationDataset
from baselines.rep_dataset import RepDataset
from src.utils import normalize_data, get_linear_results

def run_transfer_model(dataset, base_lm, transfer_lm, b=True):
    
    if dataset == "nq":
        base_dataset = OpenEndedExplanationDataset(base_lm, load_quere=True)
        transfer_dataset = OpenEndedExplanationDataset(transfer_lm, load_quere=True)
    elif dataset == "squad":
        base_dataset = SquadExplanationDataset(base_lm, load_quere=True)
        transfer_dataset = SquadExplanationDataset(transfer_lm, load_quere=True)
    
    base_train_data, base_train_labels, base_train_log_probs = \
        base_dataset.train_data, base_dataset.train_labels, base_dataset.train_log_probs
    base_train_logits, base_train_pre_conf, base_train_post_conf = base_dataset.train_logits, base_dataset.train_pre_confs, base_dataset.train_post_confs
    
    base_test_data, base_test_labels, base_test_log_probs = \
        base_dataset.test_data, base_dataset.test_labels, base_dataset.test_log_probs
    base_test_logits, base_test_pre_conf, base_test_post_conf = base_dataset.test_logits, base_dataset.test_pre_confs, base_dataset.test_post_confs

    transfer_train_data, transfer_train_labels, transfer_train_log_probs = \
        transfer_dataset.train_data, transfer_dataset.train_labels, transfer_dataset.train_log_probs
    transfer_train_logits, transfer_train_pre_conf, transfer_train_post_conf = transfer_dataset.train_logits, transfer_dataset.train_pre_confs, transfer_dataset.train_post_confs

    transfer_test_data, transfer_test_labels, transfer_test_log_probs = \
        transfer_dataset.test_data, transfer_dataset.test_labels, transfer_dataset.test_log_probs
    transfer_test_logits, transfer_test_pre_conf, transfer_test_post_conf = transfer_dataset.test_logits, transfer_dataset.test_pre_confs, transfer_dataset.test_post_confs


    results = {
        "logprob_auroc": [],
        "logits_auroc": [],
        "preconf_auroc": [],
        "postconf_auroc": [],
        "exp_auroc": [],
        "exp_all_auroc": [],
        "transfer_logprob_auroc": [],
        "transfer_logits_auroc": [],
        "transfer_preconf_auroc": [],
        "transfer_postconf_auroc": [],
        "transfer_exp_auroc": [],
        "transfer_exp_all_auroc": []
    }

    seeds = range(5)

    # unsqueeze 2nd dim of 1d outputs
    base_train_pre_conf = base_train_pre_conf.reshape(-1, 1)
    base_test_pre_conf = base_test_pre_conf.reshape(-1, 1)
    base_train_post_conf = base_train_post_conf.reshape(-1, 1)
    base_test_post_conf = base_test_post_conf.reshape(-1, 1)
    base_train_log_probs = base_train_log_probs.reshape(-1, 1)
    base_test_log_probs = base_test_log_probs.reshape(-1, 1)

    transfer_train_pre_conf = transfer_train_pre_conf.reshape(-1, 1)
    transfer_test_pre_conf = transfer_test_pre_conf.reshape(-1, 1)
    transfer_train_post_conf = transfer_train_post_conf.reshape(-1, 1)
    transfer_test_post_conf = transfer_test_post_conf.reshape(-1, 1)
    transfer_train_log_probs = transfer_train_log_probs.reshape(-1, 1)
    transfer_test_log_probs = transfer_test_log_probs.reshape(-1, 1)

    # standard z-score normalize all data with train mean and std
    base_train_data, base_test_data = normalize_data(base_train_data, base_test_data)
    base_train_log_probs, base_test_log_probs = normalize_data(base_train_log_probs, base_test_log_probs)
    base_train_pre_conf, base_test_pre_conf = normalize_data(base_train_pre_conf, base_test_pre_conf)
    base_train_post_conf, base_test_post_conf = normalize_data(base_train_post_conf, base_test_post_conf)
    base_train_logits, base_test_logits = normalize_data(base_train_logits, base_test_logits)

    transfer_train_data, transfer_test_data = normalize_data(transfer_train_data, transfer_test_data)
    transfer_train_log_probs, transfer_test_log_probs = normalize_data(transfer_train_log_probs, transfer_test_log_probs)
    transfer_train_pre_conf, transfer_test_pre_conf = normalize_data(transfer_train_pre_conf, transfer_test_pre_conf)
    transfer_train_post_conf, transfer_test_post_conf = normalize_data(transfer_train_post_conf, transfer_test_post_conf)
    transfer_train_logits, transfer_test_logits = normalize_data(transfer_train_logits, transfer_test_logits)

    for seed in seeds:
    
        # set random seed   
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # get results for logprob
        acc, f1, ece, auroc = get_linear_results(base_train_log_probs, base_train_labels, transfer_test_log_probs, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_logprob_auroc"].append(auroc)
        # get base result logprob -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_log_probs, transfer_train_labels, transfer_test_log_probs, transfer_test_labels, seed=seed, balanced=b)
        results["logprob_auroc"].append(auroc)

        # get results for preconf
        acc, f1, ece, auroc = get_linear_results(base_train_pre_conf, base_train_labels, transfer_test_pre_conf, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_preconf_auroc"].append(auroc)
        # get base result preconf -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_pre_conf, transfer_train_labels, transfer_test_pre_conf, transfer_test_labels, seed=seed, balanced=b)
        results["preconf_auroc"].append(auroc)

        # get results for postconf
        acc, f1, ece, auroc = get_linear_results(base_train_post_conf, base_train_labels, transfer_test_post_conf, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_postconf_auroc"].append(auroc)
        # get base result postconf -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_post_conf, transfer_train_labels, transfer_test_post_conf, transfer_test_labels, seed=seed, balanced=b)
        results["postconf_auroc"].append(auroc)

        # get results for logits
        acc, f1, ece, auroc = get_linear_results(base_train_logits, base_train_labels, transfer_test_logits, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_logits_auroc"].append(auroc)
        # get base result logits -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_logits, transfer_train_labels, transfer_test_logits, transfer_test_labels, seed=seed, balanced=b)
        results["logits_auroc"].append(auroc)

        # get results for exp
        acc, f1, ece, auroc = get_linear_results(base_train_data, base_train_labels, transfer_test_data, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_exp_auroc"].append(auroc)
        # get base result exp -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_data, transfer_train_labels, transfer_test_data, transfer_test_labels, seed=seed, balanced=b)
        results["exp_auroc"].append(auroc)

        # get reuslts for exp_all
        base_train_data_all = np.concatenate([base_train_data, base_train_log_probs, base_train_pre_conf, base_train_post_conf], axis=1)
        transfer_test_data_all = np.concatenate([transfer_test_data, transfer_test_log_probs, transfer_test_pre_conf, transfer_test_post_conf], axis=1)

        acc, f1, ece, auroc = get_linear_results(base_train_data_all, base_train_labels, transfer_test_data_all, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_exp_all_auroc"].append(auroc)
        # get base result exp_all -> train and test with transfer data
        transfer_train_data_all = np.concatenate([transfer_train_data, transfer_train_log_probs, transfer_train_pre_conf, transfer_train_post_conf], axis=1)
        acc, f1, ece, auroc = get_linear_results(transfer_train_data_all, transfer_train_labels, transfer_test_data_all, transfer_test_labels, seed=seed, balanced=b)
        results["exp_all_auroc"].append(auroc)

    # compute means
    results = {k: np.mean(v) for k, v in results.items()}
    results = {k: round(v, 4) for k, v in results.items()}
    # for k in ["logits_f1", "logprob_f1", "preconf_f1", "postconf_f1", "exp_f1", "exp_all_f1"]:
    for k in ["logits_auroc", "logprob_auroc", "preconf_auroc", "postconf_auroc", "exp_auroc", "exp_all_auroc", "transfer_logprob_auroc", "transfer_logits_auroc", "transfer_preconf_auroc", "transfer_postconf_auroc", "transfer_exp_auroc", "transfer_exp_all_auroc"]:
        print(k, results[k])    

def run_transfer_dataset(dataset_base, dataset_transfer, llm, b=True):

    if dataset_base == "nq":
        base_dataset = OpenEndedExplanationDataset(llm)
    elif dataset_base == "squad":
        base_dataset = SquadExplanationDataset(llm)
    elif dataset_base == "BooIQ":    
        base_dataset = ClosedEndedExplanationDataset("BooIQ", llm, load_quere=True)
    elif dataset_base == "CommonsenseQA":
        base_dataset = ClosedEndedExplanationDataset("CommonsenseQA", load_quere=True)
    elif dataset_base == "HaluEval":
        base_dataset = ClosedEndedExplanationDataset("HaluEval", load_quere=True)
    elif dataset_base == "ToxicEval":
        base_dataset = ClosedEndedExplanationDataset("ToxicEval", load_quere=True)

    if dataset_transfer == "nq":
        transfer_dataset = OpenEndedExplanationDataset(llm)
    elif dataset_transfer == "squad":
        transfer_dataset = SquadExplanationDataset(llm)
    elif dataset_transfer == "BooIQ":
        transfer_dataset = ClosedEndedExplanationDataset("BooIQ", load_quere=True)
    elif dataset_transfer == "CommonsenseQA":
        transfer_dataset = ClosedEndedExplanationDataset("CommonsenseQA", load_quere=True)
    elif dataset_transfer == "HaluEval":
        transfer_dataset = ClosedEndedExplanationDataset("HaluEval", load_quere=True)
    elif dataset_transfer == "ToxicEval":
        transfer_dataset = ClosedEndedExplanationDataset("ToxicEval", load_quere=True)

    # load base and transfer reps
    base_rep_dataset = RepDataset(dataset_base, llm)    
    transfer_rep_dataset = RepDataset(dataset_transfer, llm)

    base_train_rep = base_rep_dataset.train_rep
    base_test_rep = base_rep_dataset.test_rep

    transfer_train_rep = transfer_rep_dataset.train_rep
    transfer_test_rep = transfer_rep_dataset.test_rep
    
    base_train_data, base_train_labels, base_train_log_probs = \
        base_dataset.train_data, base_dataset.train_labels, base_dataset.train_log_probs
    base_train_logits, base_train_pre_conf, base_train_post_conf = base_dataset.train_logits, base_dataset.train_pre_confs, base_dataset.train_post_confs
    
    base_test_data, base_test_labels, base_test_log_probs = \
        base_dataset.test_data, base_dataset.test_labels, base_dataset.test_log_probs
    base_test_logits, base_test_pre_conf, base_test_post_conf = base_dataset.test_logits, base_dataset.test_pre_confs, base_dataset.test_post_confs

    transfer_train_data, transfer_train_labels, transfer_train_log_probs = \
        transfer_dataset.train_data, transfer_dataset.train_labels, transfer_dataset.train_log_probs
    transfer_train_logits, transfer_train_pre_conf, transfer_train_post_conf = transfer_dataset.train_logits, transfer_dataset.train_pre_confs, transfer_dataset.train_post_confs

    transfer_test_data, transfer_test_labels, transfer_test_log_probs = \
        transfer_dataset.test_data, transfer_dataset.test_labels, transfer_dataset.test_log_probs
    transfer_test_logits, transfer_test_pre_conf, transfer_test_post_conf = transfer_dataset.test_logits, transfer_dataset.test_pre_confs, transfer_dataset.test_post_confs

    results = {
        "logprob_auroc": [],
        "logits_auroc": [],
        "preconf_auroc": [],
        "postconf_auroc": [],
        "exp_auroc": [],
        "exp_all_auroc": [],
        "transfer_logprob_auroc": [],
        "transfer_logits_auroc": [],
        "transfer_preconf_auroc": [],
        "transfer_postconf_auroc": [],
        "transfer_exp_auroc": [],
        "transfer_exp_all_auroc": [],
        "rep_auroc": [],
        "transfer_rep_auroc": []
    }

    seeds = range(5)

    # unsqueeze 2nd dim of 1d outputs
    base_train_pre_conf = base_train_pre_conf.reshape(-1, 1)
    base_test_pre_conf = base_test_pre_conf.reshape(-1, 1)
    base_train_post_conf = base_train_post_conf.reshape(base_train_labels.shape[0], -1)
    base_test_post_conf = base_test_post_conf.reshape(base_test_labels.shape[0], -1)
    base_train_log_probs = base_train_log_probs.reshape(base_train_labels.shape[0], -1)
    base_test_log_probs = base_test_log_probs.reshape(base_test_labels.shape[0], -1)

    transfer_train_pre_conf = transfer_train_pre_conf.reshape(-1, 1)
    transfer_test_pre_conf = transfer_test_pre_conf.reshape(-1, 1)
    transfer_train_post_conf = transfer_train_post_conf.reshape(transfer_train_labels.shape[0], -1)
    transfer_test_post_conf = transfer_test_post_conf.reshape(transfer_test_labels.shape[0], -1)
    transfer_train_log_probs = transfer_train_log_probs.reshape(transfer_train_labels.shape[0], -1)
    transfer_test_log_probs = transfer_test_log_probs.reshape(transfer_test_labels.shape[0], -1)

    # standard z-score normalize all data with train mean and std
    base_train_data, base_test_data = normalize_data(base_train_data, base_test_data)
    base_train_log_probs, base_test_log_probs = normalize_data(base_train_log_probs, base_test_log_probs)
    base_train_pre_conf, base_test_pre_conf = normalize_data(base_train_pre_conf, base_test_pre_conf)
    base_train_post_conf, base_test_post_conf = normalize_data(base_train_post_conf, base_test_post_conf)
    base_train_logits, base_test_logits = normalize_data(base_train_logits, base_test_logits)

    _, transfer_test_data = normalize_data(base_train_data, transfer_test_data)
    _, transfer_test_log_probs = normalize_data(base_train_log_probs, transfer_test_log_probs)
    _, transfer_test_pre_conf = normalize_data(base_train_pre_conf, transfer_test_pre_conf)
    _, transfer_test_post_conf = normalize_data(base_train_post_conf, transfer_test_post_conf)
    _, transfer_test_logits = normalize_data(base_train_logits, transfer_test_logits)

    for seed in seeds:
    
        # set random seed   
        np.random.seed(seed)
        torch.manual_seed(seed)
        
         # get results for logprob
        acc, f1, ece, auroc = get_linear_results(base_train_log_probs, base_train_labels, transfer_test_log_probs, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_logprob_auroc"].append(auroc)
        # get base result logprob -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_log_probs, transfer_train_labels, transfer_test_log_probs, transfer_test_labels, seed=seed, balanced=b)
        results["logprob_auroc"].append(auroc)

        # get results for preconf
        acc, f1, ece, auroc = get_linear_results(base_train_pre_conf, base_train_labels, transfer_test_pre_conf, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_preconf_auroc"].append(auroc)
        # get base result preconf -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_pre_conf, transfer_train_labels, transfer_test_pre_conf, transfer_test_labels, seed=seed, balanced=b)
        results["preconf_auroc"].append(auroc)

        # get results for postconf
        acc, f1, ece, auroc = get_linear_results(base_train_post_conf, base_train_labels, transfer_test_post_conf, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_postconf_auroc"].append(auroc)
        # get base result postconf -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_post_conf, transfer_train_labels, transfer_test_post_conf, transfer_test_labels, seed=seed, balanced=b)
        results["postconf_auroc"].append(auroc)

        # get results for logits
        acc, f1, ece, auroc = get_linear_results(base_train_logits, base_train_labels, transfer_test_logits, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_logits_auroc"].append(auroc)
        # get base result logits -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_logits, transfer_train_labels, transfer_test_logits, transfer_test_labels, seed=seed, balanced=b)
        results["logits_auroc"].append(auroc)

        # get results for exp
        acc, f1, ece, auroc = get_linear_results(base_train_data, base_train_labels, transfer_test_data, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_exp_auroc"].append(auroc)
        # get base result exp -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_data, transfer_train_labels, transfer_test_data, transfer_test_labels, seed=seed, balanced=b)
        results["exp_auroc"].append(auroc)

        # get reuslts for exp_all
        base_train_data_all = np.concatenate([base_train_data, base_train_log_probs, base_train_pre_conf, base_train_post_conf], axis=1)
        transfer_test_data_all = np.concatenate([transfer_test_data, transfer_test_log_probs, transfer_test_pre_conf, transfer_test_post_conf], axis=1)

        acc, f1, ece, auroc = get_linear_results(base_train_data_all, base_train_labels, transfer_test_data_all, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_exp_all_auroc"].append(auroc)
        # get base result exp_all -> train and test with transfer data
        transfer_train_data_all = np.concatenate([transfer_train_data, transfer_train_log_probs, transfer_train_pre_conf, transfer_train_post_conf], axis=1)
        acc, f1, ece, auroc = get_linear_results(transfer_train_data_all, transfer_train_labels, transfer_test_data_all, transfer_test_labels, seed=seed, balanced=b)
        results["exp_all_auroc"].append(auroc)

        # get results for rep
        acc, f1, ece, auroc = get_linear_results(base_train_rep, base_train_labels, transfer_test_rep, transfer_test_labels, seed=seed, balanced=b)
        results["transfer_rep_auroc"].append(auroc)
        # get base result rep -> train and test with transfer data
        acc, f1, ece, auroc = get_linear_results(transfer_train_rep, transfer_train_labels, transfer_test_rep, transfer_test_labels, seed=seed, balanced=b)
        results["rep_auroc"].append(auroc)

    # compute means
    results = {k: np.mean(v) for k, v in results.items()}
    results = {k: round(v, 4) for k, v in results.items()}
    # for k in ["logits_f1", "logprob_f1", "preconf_f1", "postconf_f1", "exp_f1", "exp_all_f1"]:
    for k in ["logits_auroc", "logprob_auroc", "preconf_auroc", "postconf_auroc", "exp_auroc", "exp_all_auroc", "transfer_logits_auroc", "transfer_logprob_auroc", "transfer_preconf_auroc", "transfer_postconf_auroc", "transfer_exp_auroc", "transfer_exp_all_auroc", "rep_auroc", "transfer_rep_auroc"]:
        print(k, results[k])   

if __name__ == "__main__":

    # dataset = "nq"
    dataset = "squad"
    # base_lm = "llama3-8b"
    # transfer_lm = "llama3-70b"
    base_lm = "llama3-3b"
    transfer_lm = "llama3-8b"
    run_transfer_model(dataset, base_lm, transfer_lm)

    # base_dataset = "nq"
    # transfer_dataset = "squad"
    # base_dataset = "squad"
    # transfer_dataset = "nq"

    # base_dataset = "HaluEval"
    # transfer_dataset = "ToxicEval"
    # base_dataset = "ToxicEval"
    # transfer_dataset = "HaluEval"

    # llm = "llama3-70b"
    # run_transfer_dataset(base_dataset, transfer_dataset, llm)