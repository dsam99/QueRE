import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.quere import SquadExplanationDataset
from src.quere_oai import SquadExplanationDataset_OAI
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_clusters():

    model_datasets = []
    # models = ["llama-7b", "llama-13b", "llama-70b"]
    models = ["llama3-3b", "llama3-8b", "llama3-70b"]
    models_legend = ["LLaMA3-3B", "LLaMA3-8B", "LLAMA3-70B"]
    # models = ["mistral-7b", "mistral-8x7b"]
    # models = ["gpt-3.5-turbo-0125", "gpt-4-turbo-preview"]
    # models_legend = ["GPT-3.5", "GPT-4"]
    for model in models:

        # load data 
        if "gpt" in model:
            dataset = SquadExplanationDataset_OAI(model)
            # dataset = OpenEndedExplanationDataset_OAI(model)
            # dataset = BooIQExplanationDataset_OAI("BooIQ", model)
            test_data = dataset.test_data
            d_len = len(dataset.test_labels)
            test_log_probs = dataset.test_log_probs.reshape(d_len, -1)
            test_pre_conf, test_post_conf = dataset.test_pre_confs.reshape(d_len, -1), dataset.test_post_confs.reshape(d_len, -1)
            test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)
            
            # subset only 1000 instances for visualization
            test_data_all = test_data_all[:1000]
            model_datasets.append(test_data_all)

        else:
            dataset = SquadExplanationDataset(model, load_quere=True)
            test_data = dataset.test_data
            d_len = len(dataset.test_labels)
            test_log_probs = dataset.test_log_probs.reshape(d_len, -1)
            test_pre_conf, test_post_conf = dataset.test_pre_confs.reshape(d_len, -1), dataset.test_post_confs.reshape(d_len, -1)

            test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)
            
            # subset only 1000 instances for visualization
            test_data_all = test_data_all[:1000]
            model_datasets.append(test_data_all)

        # model_datasets.append(test_data)
    
    # colors = plt.cm.get_cmap('viridis', len(models))  # Get a colormap with enough colors
    # colors = ["lightcoral", "lightcyan", "palegreen"]
    # colors = [(0.678, 0.847, 0.902), (0.858, 0.717, 0.949), (1.0, 0.713, 0.756)]
    colors = [(0.578, 0.747, 0.802), (0.758, 0.617, 0.849), (0.900, 0.613, 0.656)]


    # plot clusters of our data -> after doing TSNE onto two components
    all_test_data = np.concatenate(model_datasets, axis=0)

    # normalize data
    all_test_data = (all_test_data - np.mean(all_test_data, axis=0)) / np.std(all_test_data, axis=0)
    # tsne = TSNE(n_components=2, random_state=0, perplexity=50) # for llama
    tsne = TSNE(n_components=2, random_state=0, perplexity=10) # for oai
    tsne_results = tsne.fit_transform(all_test_data)

    # pca = PCA(n_components=2)
    # tsne_results = pca.fit_transform(all_test_data)

    # Plot each model's data with a unique color and label
    index = 0
    for i, model_data in enumerate(model_datasets):
        model_tsne = tsne_results[index:index+len(model_data)]
        plt.scatter(model_tsne[:, 0], model_tsne[:, 1], color=colors[i], label=models_legend[i], s=8)
        index += len(model_data)

    # add legend
    plt.legend(fontsize=12)
    plt.tight_layout()

    # make xticks and yticks not visible
    plt.xticks([])
    plt.yticks([])

    if "gpt" in model:

        plt.savefig("figs/clusters_oai.png")
        plt.savefig("figs/clusters_oai.pdf")

    else:
        plt.savefig("figs/clusters.png")
        plt.savefig("figs/clusters.pdf")

if __name__ == "__main__":
    # plot_clusters("squad")
    plot_clusters()