import random
from src.llm import load_llm
from src.quere import ClosedEndedExplanationDataset
from src.utils import get_linear_results
import numpy as np
import torch

def get_random_tokens():

    # for each sequence, sample 20 tokens
    tokens_per_list = 20
    
    # sample 10 sequences
    num_lists = 10

    # set random seed
    random.seed(0)

    # Generate random tokens from a range of 128000
    random_token_lists = [
        [random.randint(0, 128000) for _ in range(tokens_per_list)] for _ in range(num_lists)
    ]
    
    # Print the random token lists
    for i, token_list in enumerate(random_token_lists, 1):
        print(f"List {i}: {token_list}")
    

    _, tokenizer = load_llm("llama3-3b")
    # Decode the random tokens
    decoded_random_token_lists = [
        tokenizer.decode(token_list) for token_list in random_token_lists
    ]

    for i, decoded_token_list in enumerate(decoded_random_token_lists, 1):
        print(f"List {i}: {decoded_token_list}")

def run_random_tokens(dataset, llm):

    if dataset == "BooIQ":    
        base_dataset = ClosedEndedExplanationDataset("BooIQ", llm)
        random_dataset = ClosedEndedExplanationDataset("BooIQ", llm, random=True)
        random_tokens_dataset = ClosedEndedExplanationDataset("BooIQ", llm, random_tokens=True)
    elif dataset == "CommonsenseQA":
        base_dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm)
        random_dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, random=True)
        random_tokens_dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, random_tokens=True)
    elif dataset == "HaluEval":
        base_dataset = ClosedEndedExplanationDataset("HaluEval", llm)
        random_dataset = ClosedEndedExplanationDataset("HaluEval", llm, random=True)
        random_tokens_dataset = ClosedEndedExplanationDataset("HaluEval", llm, random_tokens=True)
    elif dataset == "ToxicEval":
        base_dataset = ClosedEndedExplanationDataset("ToxicEval", llm)
        random_dataset = ClosedEndedExplanationDataset("ToxicEval", llm, random=True)
        random_tokens_dataset = ClosedEndedExplanationDataset("ToxicEval", llm, random_tokens=True)
    
    balanced=True

    # 2. Extract train/test data and labels
    base_train_data = base_dataset.train_data
    base_train_labels = base_dataset.train_labels
    base_test_data = base_dataset.test_data
    base_test_labels = base_dataset.test_labels

    random_train_data = random_dataset.train_data
    random_test_data = random_dataset.test_data
    random_train_labels = random_dataset.train_labels
    random_test_labels = random_dataset.test_labels
    
    randtok_train_data = random_tokens_dataset.train_data
    randtok_test_data = random_tokens_dataset.test_data
    randtok_train_labels = random_tokens_dataset.train_labels
    randtok_test_labels = random_tokens_dataset.test_labels

    # 3. For demonstration, we’ll only use the “raw” data features (i.e., no logits, no log_probs)
    #    but you can adapt this to also test log_probs, confidence scores, etc.

    results = {
        "base_acc": [],
        "base_f1": [],
        "base_ece": [],
        "base_auroc": [],

        "random_acc": [],
        "random_f1": [],
        "random_ece": [],
        "random_auroc": [],

        "randtok_acc": [],
        "randtok_f1": [],
        "randtok_ece": [],
        "randtok_auroc": [],
    }

    # You can adjust the number of seeds as needed
    seeds = range(1)

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # print out shapes
        # print("base", base_train_data.shape, base_test_data.shape)
        # print("random", random_train_data.shape, random_test_data.shape)
        # print("random tokens", randtok_train_data.shape, randtok_test_data.shape)

        # --- Train on base_dataset.train_data, test on base_dataset.test_data ---
        acc, f1, ece, auroc = get_linear_results(
            base_train_data,
            base_train_labels,
            base_test_data,
            base_test_labels,
            seed=seed,
            balanced=balanced
        )
        results["base_acc"].append(acc)
        results["base_f1"].append(f1)
        results["base_ece"].append(ece)
        results["base_auroc"].append(auroc)

        # --- Train on random_dataset.train_data, test on base_dataset.test_data ---
        acc, f1, ece, auroc = get_linear_results(
            random_train_data,
            random_train_labels,
            random_test_data,
            random_test_labels,
            seed=seed,
            balanced=balanced
        )
        results["random_acc"].append(acc)
        results["random_f1"].append(f1)
        results["random_ece"].append(ece)
        results["random_auroc"].append(auroc)

        # --- Train on random_tokens_dataset.train_data, test on base_dataset.test_data ---
        acc, f1, ece, auroc = get_linear_results(
            randtok_train_data,
            randtok_train_labels,
            randtok_test_data,
            randtok_test_labels,
            seed=seed,
            balanced=balanced
        )
        results["randtok_acc"].append(acc)
        results["randtok_f1"].append(f1)
        results["randtok_ece"].append(ece)
        results["randtok_auroc"].append(auroc)

    # Aggregate (mean) the results across seeds
    results_mean = {k: np.mean(v) for k, v in results.items()}

    # Round for nicer printing
    results_rounded = {k: round(v, 4) for k, v in results_mean.items()}

    # Print the comparison
    print("Comparison of base vs. random vs. random_tokens (trained on each version, tested on base test data):")
    print("-------------------------------------------------------------------")
    for metric_group in ["auroc"]:
        b = results_rounded[f"base_{metric_group}"]
        r = results_rounded[f"random_{metric_group}"]
        rt = results_rounded[f"randtok_{metric_group}"]
        print(f"{metric_group.upper()}: base={b}, random={r}, rand_tokens={rt}")
    print("-------------------------------------------------------------------\n")



if __name__ == "__main__":
    # get_random_tokens()
    # run_random_tokens("BooIQ", "llama3-8b")
    # run_random_tokens("HaluEval", "llama3-8b")
    run_random_tokens("HaluEval", "llama3-3b")