import torch
import numpy as np
from sklearn.metrics import f1_score

from src.quere import SquadExplanationDataset, OpenEndedExplanationDataset, ClosedEndedExplanationDataset
from baselines.rep_dataset import RepDataset
from src.utils import train_linear_model, compute_ece
from src.llm import load_llm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd


def get_f1_thresh(probs, labels, threshs):
    f1s = []
    for thresh in threshs:
        confs = np.abs(probs - 0.5)
        pred_inds = np.where(confs > thresh)[0]
        y_pred = (probs > 0.5).astype(int) 
        f1 = f1_score(labels[pred_inds], y_pred[pred_inds])
        f1s.append(f1)
    return np.array(f1s)

def get_acc_thresh(probs, labels, threshs):
    accs = []
    for thresh in threshs:
        confs = np.abs(probs - 0.5)
        pred_inds = np.where(confs > thresh)[0]
        y_pred = (probs > 0.5).astype(int) 
        acc = (y_pred[pred_inds] == labels[pred_inds]).mean()
        accs.append(acc)
    return np.array(accs)

def get_acc_bins(probs, labels, nbins=10):
    # confs = [1 - p if p < 0.5 else p for p in probs]
    confs = probs
    df = pd.DataFrame({'probs': probs, 'labels': labels, 'confs': confs})
    df['bin'], bin_edges = pd.qcut(df['confs'], q=nbins, labels=False, retbins=True, duplicates='drop')
    bin_centers = []
    accuracies = []
    counts = []

    for i in range(df['bin'].nunique()):
        bin_df = df[df['bin'] == i]
        if len(bin_df) == 0:
            bin_centers.append(np.nan)
            accuracies.append(np.nan)
            counts.append(0)
        else:
            acc = ((bin_df['probs'] > 0.5).astype(int) == bin_df['labels']).mean()
            accuracies.append(acc)
            bin_centers.append(bin_df['confs'].mean())
            counts.append(len(bin_df))

    return bin_centers, accuracies, counts

def plot_accuracy_vs_confidence(dataset_name, models):
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    # Initialize plot
    plt.figure(figsize=(10, 6))
    
    for idx, (llm, color) in enumerate(models):
        # Load the dataset
        if dataset_name == "BooIQ":
            dataset = ClosedEndedExplanationDataset("BooIQ", llm, load_quere=True)
        elif dataset_name == "HaluEval":
            dataset = ClosedEndedExplanationDataset("HaluEval", llm, load_quere=True)
        elif dataset_name == "ToxicEval":
            dataset = ClosedEndedExplanationDataset("ToxicEval", llm, load_quere=True)    
        elif dataset_name == "CommonsenseQA":
            dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, load_quere=True)
        elif dataset_name == "WinoGrande":
            dataset = ClosedEndedExplanationDataset("WinoGrande", llm, load_quere=True)
        elif dataset_name == "squad":
            dataset = SquadExplanationDataset(llm, load_quere=True)
        elif dataset_name == "nq":
            dataset = OpenEndedExplanationDataset(llm, load_quere=True)    


        b = True  # For balanced training
    
        rep_dataset = RepDataset(dataset_name, llm)
        train_rep = rep_dataset.train_rep
        test_rep = rep_dataset.test_rep
    
        train_data, train_labels, train_log_probs = \
            dataset.train_data, dataset.train_labels, dataset.train_log_probs
        
        test_data, test_labels, test_log_probs, = \
            dataset.test_data, dataset.test_labels, dataset.test_log_probs
        train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
        test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs
    
        train_pre_conf = train_pre_conf.reshape(train_labels.shape[0], -1)
        test_pre_conf = test_pre_conf.reshape(test_labels.shape[0], -1)
        train_post_conf = train_post_conf.reshape(train_labels.shape[0], -1)
        test_post_conf = test_post_conf.reshape(test_labels.shape[0], -1)
        train_log_probs = train_log_probs.reshape(train_labels.shape[0], -1)
        test_log_probs = test_log_probs.reshape(test_labels.shape[0], -1)
    
        # Train and predict using the answer probabilities
        answer_clf = train_linear_model(train_log_probs, train_labels, test_log_probs, test_labels, seed=0, balanced=b)
        y_prob_ans = answer_clf.predict_proba(test_log_probs)[:, 1]
        y_pred_ans = (y_prob_ans > 0.5).astype(int)
        # bin_centers_ans, accuracies_ans, counts_ans = get_acc_bins(y_prob_ans, test_labels, nbins=5)

        y_prob_ans_rev = 1 - y_prob_ans


        # combine the two predictions
        y_probs_total = np.concatenate([y_prob_ans, y_prob_ans_rev])
        y_labels_total = np.concatenate([test_labels, test_labels])
        
        bin_centers_ans, accuracies_ans, counts_ans = get_acc_bins(y_probs_total, y_labels_total, nbins=10)

        # Train and predict using QueRE method
        train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
        test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)
    
        our_clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels, seed=0, balanced=b)
        y_prob_our = our_clf.predict_proba(test_data_all)[:, 1]
        y_pred_our = (y_prob_our > 0.5).astype(int)
        # bin_centers_our, accuracies_our, counts_our = get_acc_bins(y_prob_our, test_labels, nbins=5)
    
        y_prob_our_rev = 1 - y_prob_our
        bin_centers_our, accuracies_our, counts_our = get_acc_bins(np.concatenate([y_prob_our, y_prob_our_rev]), np.concatenate([test_labels, test_labels]), nbins=10)

        # Plotting for Answer Probs (dashed line)
        plt.plot(bin_centers_ans, accuracies_ans, label=f'{llm} Answer Probs', color=color, linestyle='--', linewidth=2.5)
        # Plotting for QueRE (solid line)
        plt.plot(bin_centers_our, accuracies_our, label=f'{llm} QueRE', color=color, linestyle='-', linewidth=2.5)
    
    # Plot y = x line (Best possible performance)
    plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=1)
    
    plt.xlabel('Average Confidence in Bin', fontsize=24)
    plt.ylabel('Accuracy', fontsize=24)
    # plt.title(f'Accuracy vs. Confidence Bins for {dataset_name}', fontsize=24)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figs/calibration/binned_accuracy_{dataset_name}.png")
    plt.savefig(f"figs/calibration/binned_accuracy_{dataset_name}.pdf")
    plt.close()


def compute_ece(probs, labels, nbins=10):
    """
    Computes the Expected Calibration Error (ECE).

    Parameters:
    - probs: Predicted probabilities for the positive class.
    - labels: True binary labels.
    - nbins: Number of bins to use for calibration.

    Returns:
    - ece: The expected calibration error.
    """
    bin_edges = np.linspace(0, 1, nbins + 1)
    bin_indices = np.digitize(probs, bin_edges, right=True) - 1  # Bin indices start from 0
    ece = 0.0
    total_samples = len(probs)

    for i in range(nbins):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_probs = probs[bin_mask]
            bin_labels = labels[bin_mask]
            bin_confidence = np.mean(bin_probs)
            bin_accuracy = np.mean(bin_labels)
            bin_error = np.abs(bin_confidence - bin_accuracy)
            ece += (bin_size / total_samples) * bin_error

    return ece

def compute_and_plot_ece(dataset_name, models):
    # Set random seed
    np.random.seed(0)
    torch.manual_seed(0)
    
    methods = ['Answer Probs', 'QueRE (Ours)']
    ece_results = {method: [] for method in methods}
    model_names = []

    for idx, (llm, color) in enumerate(models):
        # Load the dataset
        if dataset_name == "BooIQ":
            dataset = ClosedEndedExplanationDataset("BooIQ", llm, load_quere=True)
        elif dataset_name == "HaluEval":
            dataset = ClosedEndedExplanationDataset("HaluEval", llm, load_quere=True)
        elif dataset_name == "ToxicEval":
            dataset = ClosedEndedExplanationDataset("ToxicEval", llm, load_quere=True)    
        elif dataset_name == "CommonsenseQA":
            dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, load_quere=True)
        elif dataset_name == "WinoGrande":
            dataset = ClosedEndedExplanationDataset("WinoGrande", llm, load_quere=True)
        elif dataset_name == "squad":
            dataset = SquadExplanationDataset(llm, load_quere=True)
        elif dataset_name == "nq":
            dataset = OpenEndedExplanationDataset(llm, load_quere=True)    

        b = True  # For balanced training
    
        rep_dataset = RepDataset(dataset_name, llm)
        train_rep = rep_dataset.train_rep
        test_rep = rep_dataset.test_rep
    
        train_data, train_labels, train_log_probs = \
            dataset.train_data, dataset.train_labels, dataset.train_log_probs
        
        test_data, test_labels, test_log_probs = \
            dataset.test_data, dataset.test_labels, dataset.test_log_probs
        train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
        test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs
    
        train_pre_conf = train_pre_conf.reshape(train_labels.shape[0], -1)
        test_pre_conf = test_pre_conf.reshape(test_labels.shape[0], -1)
        train_post_conf = train_post_conf.reshape(train_labels.shape[0], -1)
        test_post_conf = test_post_conf.reshape(test_labels.shape[0], -1)
        train_log_probs = train_log_probs.reshape(train_labels.shape[0], -1)
        test_log_probs = test_log_probs.reshape(test_labels.shape[0], -1)
    
        # clip train and test log probs
        if dataset_name == "nq":
            train_log_probs = np.clip(train_log_probs, -10000, 0)
            test_log_probs = np.clip(test_log_probs, -10000, 0)

        # Train and predict using the answer probabilities
        answer_clf = train_linear_model(train_log_probs, train_labels, test_log_probs, test_labels, seed=0, balanced=b)
        y_prob_ans = answer_clf.predict_proba(test_log_probs)[:, 1]
        y_pred_ans = (y_prob_ans >= 0.5).astype(int)

        # Compute ECE for Answer Probs
        ece_ans = compute_ece(y_prob_ans, test_labels, nbins=10)
        ece_results['Answer Probs'].append(ece_ans)
    
        # Train and predict using QueRE method
        train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
        test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)
    
        our_clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels, seed=0, balanced=b)
        y_prob_our = our_clf.predict_proba(test_data_all)[:, 1]
        y_pred_our = (y_prob_our >= 0.5).astype(int)

        # Compute ECE for QueRE
        ece_our = compute_ece(y_prob_our, test_labels, nbins=10)
        ece_results['QueRE (Ours)'].append(ece_our)

        # Collect model names
        model_names.append(llm)
    
    # Plotting ECE Bar Chart
    x = np.arange(len(model_names))  # Label locations
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Bars for Answer Probs
    rects1 = ax.bar(x - width/2, ece_results['Answer Probs'], width, label='Answer Probs', color='skyblue', edgecolor='black')
    # Bars for QueRE
    rects2 = ax.bar(x + width/2, ece_results['QueRE (Ours)'], width, label='QueRE (Ours)', color='steelblue', edgecolor='black')

    # change name mapping
    model_names = [m.replace("llama3-", "LLaMA3-") for m in model_names]
    model_names = [m.replace("mistral-", "Mistral-") for m in model_names]
    model_names = [m.replace("b", "B") for m in model_names]


    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel('ECE', fontsize=32)
    plt.yticks(fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=28)
    
    if dataset_name == "HaluEval" or dataset_name == "WinoGrande":
        ax.legend(fontsize=20)
    


    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Attach a text label above each bar displaying its height
    def autolabel(rects):
        """Attach a text label displaying the height of each bar."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),  # Offset label position
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig(f"figs/ece/ece_{dataset_name}.png")
    plt.savefig(f"figs/ece/ece_{dataset_name}.pdf")
    plt.close()

if __name__ == "__main__":

    # set random seed
    # np.random.seed(0)
    # torch.manual_seed(0)
    # vary_pred_threshold_gpt("squad", "llama-70b")
    # vary_pred_threshold_gpt("squad", "mistral-8x7b")

    models = [
        # ("llama3-3b", '#CD5C5C'),
        ("llama3-8b", '#4682B4'),
        ("llama3-70b", '#6B8E23'),
        # You can add more models and assign colors
    ]
    # for d in ["HaluEval", "ToxicEval", "CommonsenseQA", "WinoGrande", "squad", "nq"]:
    for d in ["BooIQ"]:
        # plot_accuracy_vs_confidence(d, models)
        compute_and_plot_ece(d, models)