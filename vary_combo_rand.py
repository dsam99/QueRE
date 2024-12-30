import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.quere import ClosedEndedExplanationDataset, SquadExplanationDataset, OpenEndedExplanationDataset
from baselines.rep_dataset import RepDataset
from src.utils import train_linear_model, compute_ece, get_linear_results

def vary_number_prompts_random(dataset_name, llm):

	if dataset_name == "BooIQ":
		dataset = ClosedEndedExplanationDataset("BooIQ", llm)
	elif dataset_name == "HaluEval":
		dataset = ClosedEndedExplanationDataset("HaluEval", llm)
	elif dataset_name == "ToxicEval":
		dataset = ClosedEndedExplanationDataset("ToxicEval", llm)    
	elif dataset_name == "CommonsenseQA":
		dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm)
	elif dataset_name == "WinoGrande":
		dataset = ClosedEndedExplanationDataset(llm)
	elif dataset_name == "squad":
		dataset = SquadExplanationDataset(llm)
	elif dataset_name == "nq":
		dataset = OpenEndedExplanationDataset(llm)	
	# load random dataset
	if dataset_name == "BooIQ":
		random_dataset = ClosedEndedExplanationDataset("BooIQ", llm, random=True)
	elif dataset_name == "HaluEval":
		random_dataset = ClosedEndedExplanationDataset("HaluEval", llm, random=True)
	elif dataset_name == "ToxicEval":
		random_dataset = ClosedEndedExplanationDataset("ToxicEval", llm, random=True)
	elif dataset_name == "CommonsenseQA":
		random_dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, random=True)
	elif dataset_name == "WinoGrande":
		random_dataset = ClosedEndedExplanationDataset(llm, random=True)
	elif dataset_name == "squad":
		random_dataset = SquadExplanationDataset(llm, random=True)
	elif dataset_name == "nq":
		random_dataset = OpenEndedExplanationDataset(llm, random=True)
	
	b = True

	train_data, train_labels, train_log_probs = \
		dataset.train_data, dataset.train_labels, dataset.train_log_probs
	
	test_data, test_labels, test_log_probs, = \
		dataset.test_data, dataset.test_labels, dataset.test_log_probs
	
	train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
	test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

	# reshape pre_conf, post_conf, log_probs
	train_pre_conf = train_pre_conf.reshape(len(train_data), -1)
	test_pre_conf = test_pre_conf.reshape(len(test_data), -1)
	train_post_conf = train_post_conf.reshape(len(train_data), -1)
	test_post_conf = test_post_conf.reshape(len(test_data), -1)

	train_log_probs = train_log_probs.reshape(len(train_data), -1)
	test_log_probs = test_log_probs.reshape(len(test_data), -1)


	train_random_data = random_dataset.train_data
	test_random_data = random_dataset.test_data

	print("train_random_data", train_random_data.shape)
	print("train_data", train_data.shape)

	seeds = range(20)
	num_prompt_list = range(2, 8, 1)
	
	# train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
	# test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

	train_data_all = train_data
	test_data_all = test_data

	print(train_data.shape, test_data.shape)

	results = {}
	random_results = {}
	for s in num_prompt_list: # number of random used out of 10
		results[s] = []

	for seed in seeds:
		# set random seed
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		# randomly shuffle a list of inds to select
		random_idxs = np.random.permutation(train_data.shape[1])
		norm_idxs = np.random.permutation(train_data.shape[1])

		for num_prompts in num_prompt_list:
			random_ids = random_idxs[:num_prompts]
			norm_ids = norm_idxs[:num_prompts]

			train_data_subset = train_data_all[:, norm_ids]
			test_data_subset = test_data_all[:, norm_ids]

			train_random_data_subset = train_random_data[:, random_ids]
			test_random_data_subset = test_random_data[:, random_ids]

			all_train_data = np.concatenate([train_data_subset, train_random_data_subset], axis=1)
			all_test_data = np.concatenate([test_data_subset, test_random_data_subset], axis=1)

			# train predictor
			acc, f1, ece, auroc = get_linear_results(all_train_data, train_labels, all_test_data, test_labels, seed=seed, balanced=b)
			results[num_prompts].append(auroc)

	# plot results
	import matplotlib.pyplot as plt
	plt.figure()

	# average over seeds and compute std
	means = []
	stds = []

	for s in num_prompt_list:
		means.append(np.mean(results[s]))
		stds.append(np.std(results[s]) / np.sqrt(len(seeds)))

	print(means, stds)
	
	# plt.errorbar(num_prompt_list, means, yerr=stds)
	plt.plot(num_prompt_list, means, label="Combination of Elicitation Prompts and Random Sequences")
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2)
	plt.xlabel("Number of Random Sequences", fontsize=24)

	# set tick size
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("AUROC", fontsize=24)

	plt.legend()

	# tight layout
	plt.tight_layout()

	plt.savefig("figs/combo_" + dataset_name + "_" + llm + ".png")
	plt.savefig("figs/combo_" + dataset_name + "_" + llm + ".pdf")


if __name__ == "__main__":

	# set random seed
	np.random.seed(0)
	torch.manual_seed(0)

	llm = "llama3-70b"
	vary_number_prompts_random("squad", llm)