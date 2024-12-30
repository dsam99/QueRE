import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.quere import ClosedEndedExplanationDataset, SquadExplanationDataset, OpenEndedExplanationDataset
from baselines.rep_dataset import RepDataset
import sys
import argparse
from tqdm import tqdm
from src.utils import train_linear_model, compute_ece, get_linear_results


def vary_train_data(dataset_name, llm):
	
	# set random seed
	np.random.seed(0)
	torch.manual_seed(0)

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

	b = True

	rep_dataset = RepDataset(dataset_name, llm)
	train_rep = rep_dataset.train_rep
	test_rep = rep_dataset.test_rep

	train_data, train_labels, train_log_probs = \
		dataset.train_data, dataset.train_labels, dataset.train_log_probs
	
	test_data, test_labels, test_log_probs, = \
		dataset.test_data, dataset.test_labels, dataset.test_log_probs
	train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
	test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

	train_pre_conf = train_pre_conf.reshape(-1, 1)
	test_pre_conf = test_pre_conf.reshape(-1, 1)

	all_train_data = train_data.copy()
	all_train_labels = train_labels.copy()
	all_train_log_probs = train_log_probs.copy()
	all_train_logits = train_logits.copy()
	all_train_pre_conf = train_pre_conf.copy()
	all_train_post_conf = train_post_conf.copy()
	all_train_rep = train_rep.copy()

	train_amts = [20, 50, 100, 250, 500, 750, 1000]
	means = np.zeros((len(train_amts), 7))
	stds = np.zeros((len(train_amts), 7))

	for amt in tqdm(train_amts, total=len(train_amts)):

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
			"rep_acc": [],
			"rep_f1": [],
			"rep_ece": [],
			"logprob_auroc": [],
			"logits_auroc": [],
			"preconf_auroc": [],
			"postconf_auroc": [],
			"exp_auroc": [],    
			"exp_all_auroc": [],
			"rep_auroc": [],
		}

		seeds = range(10)
		
		for seed in seeds:
		
			# set random seed   
			np.random.seed(seed)
			torch.manual_seed(seed)

			# randomly shuffle and select indices
			idxs = np.random.permutation(len(all_train_data))[:amt]

			train_data = all_train_data[idxs]
			train_labels = all_train_labels[idxs]
			train_log_probs = all_train_log_probs[idxs]
			train_logits = all_train_logits[idxs]
			train_pre_conf = all_train_pre_conf[idxs]
			train_post_conf = all_train_post_conf[idxs]
			train_rep = all_train_rep[idxs]

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
		for i, k in enumerate(["logits_auroc", "rep_auroc", "logprob_auroc", "preconf_auroc", "postconf_auroc", "exp_auroc", "exp_all_auroc"]):
			means[train_amts.index(amt), i] = results[k]
			stds[train_amts.index(amt), i] = results[k]

	# plot results - only means
	import matplotlib.pyplot as plt
	plt.figure()
	plt.plot(train_amts, means[:, 0], label="logits")
	plt.plot(train_amts, means[:, 1], label="rep") 
	plt.plot(train_amts, means[:, 2], label="logprob")
	plt.plot(train_amts, means[:, 3], label="preconf")
	plt.plot(train_amts, means[:, 4], label="postconf")
	plt.plot(train_amts, means[:, 5], label="exp")
	plt.plot(train_amts, means[:, 6], label="exp_all")
	plt.legend()
	# plt.show()
	plt.savefig("figs/vary_train_data_" + dataset_name + "_" + llm + ".png")

def vary_number_prompts(dataset_name, llm):

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

	b = True

	train_data, train_labels, train_log_probs = \
		dataset.train_data, dataset.train_labels, dataset.train_log_probs
	
	test_data, test_labels, test_log_probs, = \
		dataset.test_data, dataset.test_labels, dataset.test_log_probs
	
	seeds = range(10)
	num_prompt_list = range(5, 50, 5)
	
	print(train_data.shape, test_data.shape)

	results = {}
	for s in num_prompt_list:
		results[s] = []

	for seed in seeds:
		# set random seed
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		# randomly shuffle a list of inds to select
		idxs = np.random.permutation(train_data.shape[1])

		for num_prompts in num_prompt_list:
			prompt_ids = idxs[:num_prompts]

			train_data_subset = train_data[:, prompt_ids]
			test_data_subset = test_data[:, prompt_ids]

			# train predictor
			acc, f1, ece, auroc = get_linear_results(train_data_subset, train_labels, test_data_subset, test_labels, seed=seed, balanced=b)
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
	plt.plot(num_prompt_list, means)
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2)
	plt.xlabel("Number of Elicitation Prompts", fontsize=24)

	# set tick size
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("AUROC", fontsize=24)

	# tight layout
	plt.tight_layout()

	plt.savefig("figs/vary_num_prompts_" + dataset_name + "_" + llm + ".png")
	plt.savefig("figs/vary_num_prompts_" + dataset_name + "_" + llm + ".pdf")


def vary_number_prompts_random(dataset_name, llm):

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
	num_prompt_list = range(2, 11, 2)
	
	train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
	test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

	print(train_data.shape, test_data.shape)

	results = {}
	random_results = {}
	for s in num_prompt_list:
		results[s] = []
		random_results[s] = []

	for seed in seeds:
		# set random seed
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		# randomly shuffle a list of inds to select
		idxs = np.random.permutation(train_random_data.shape[1])

		for num_prompts in num_prompt_list:
			prompt_ids = idxs[:num_prompts]

			train_data_subset = train_data_all[:, prompt_ids]
			test_data_subset = test_data_all[:, prompt_ids]

			train_random_data_subset = train_random_data[:, prompt_ids]
			test_random_data_subset = test_random_data[:, prompt_ids]

			# add in pre, post conf, answer probs, logits
			# train_data_subset_all = np.concatenate([train_data_subset, train_log_probs, train_pre_conf, train_post_conf], axis=1)
			# test_data_subset_all = np.concatenate([test_data_subset, test_log_probs, test_pre_conf, test_post_conf], axis=1)

			# train_random_data_subset_all = np.concatenate([train_random_data_subset, train_log_probs, train_pre_conf, train_post_conf], axis=1)
			# test_random_data_subset_all = np.concatenate([test_random_data_subset, test_log_probs, test_pre_conf, test_post_conf], axis=1)

			# train predictor
			acc, f1, ece, auroc = get_linear_results(train_data_subset, train_labels, test_data_subset, test_labels, seed=seed, balanced=b)
			results[num_prompts].append(auroc)

			# train random predictor
			acc, f1, ece, auroc = get_linear_results(train_random_data_subset, train_labels, test_random_data_subset, test_labels, seed=seed, balanced=b)
			random_results[num_prompts].append(auroc)

	# plot results
	import matplotlib.pyplot as plt
	plt.figure()

	# average over seeds and compute std
	means = []
	stds = []

	random_means = []
	random_stds = []

	for s in num_prompt_list:
		means.append(np.mean(results[s]))
		stds.append(np.std(results[s]) / np.sqrt(len(seeds)))

		random_means.append(np.mean(random_results[s]))
		random_stds.append(np.std(random_results[s]) / np.sqrt(len(seeds)))

	print(means, stds)
	print(random_means, random_stds)

	print(means, random_means)
	
	# plt.errorbar(num_prompt_list, means, yerr=stds)
	plt.plot(num_prompt_list, means, label="QueRE")
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2)
	plt.xlabel("Number of Elicitation Prompts", fontsize=24)

	plt.plot(num_prompt_list, random_means, label="Random")
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(random_means, random_stds)], [m + s for m, s in zip(random_means, random_stds)], alpha=0.2)

	# set tick size
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("AUROC", fontsize=24)

	plt.legend()

	# tight layout
	plt.tight_layout()

	plt.savefig("figs/vary_num_prompts_random_" + dataset_name + "_" + llm + ".png")
	plt.savefig("figs/vary_num_prompts_random_" + dataset_name + "_" + llm + ".pdf")


def vary_number_prompts_diverse(dataset_name, llm):


	if dataset_name == "BooIQ":
		dataset = ClosedEndedExplanationDataset("BooIQ", llm, gpt_exp=True)
	elif dataset_name == "HaluEval":
		dataset = ClosedEndedExplanationDataset("HaluEval", llm, gpt_exp=True)
	elif dataset_name == "ToxicEval":
		dataset = ClosedEndedExplanationDataset("ToxicEval", llm, gpt_exp=True)
	elif dataset_name == "CommonsenseQA":
		dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, gpt_exp=True)
	elif dataset_name == "WinoGrande":
		dataset = ClosedEndedExplanationDataset("WinoGrande", llm, gpt_exp=True)
	elif dataset_name == "squad":
		dataset = SquadExplanationDataset(llm, gpt_exp=True)
	elif dataset_name == "nq":
		dataset = OpenEndedExplanationDataset(llm, gpt_exp=True)
	
	# load diverse dataset
	if dataset_name == "BooIQ":
		random_dataset = ClosedEndedExplanationDataset("BooIQ", llm, gpt_diverse=True)
	elif dataset_name == "HaluEval":
		random_dataset = ClosedEndedExplanationDataset("HaluEval", llm, gpt_diverse=True)
	elif dataset_name == "ToxicEval":
		random_dataset = ClosedEndedExplanationDataset("ToxicEval", llm, gpt_diverse=True)
	elif dataset_name == "CommonsenseQA":
		random_dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, gpt_diverse=True)
	elif dataset_name == "WinoGrande":
		random_dataset = ClosedEndedExplanationDataset(llm, gpt_diverse=True)
	elif dataset_name == "squad":
		random_dataset = SquadExplanationDataset(llm, gpt_diverse=True)
	elif dataset_name == "nq":
		random_dataset = OpenEndedExplanationDataset(llm, gpt_diverse=True)

	# load similar dataset
	if dataset_name == "BooIQ":
		sim_dataset = ClosedEndedExplanationDataset("BooIQ", llm, gpt_sim=True)
	elif dataset_name == "HaluEval":
		sim_dataset = ClosedEndedExplanationDataset("HaluEval", llm, gpt_sim=True)
	elif dataset_name == "ToxicEval":
		sim_dataset = ClosedEndedExplanationDataset("ToxicEval", llm, gpt_sim=True)
	elif dataset_name == "CommonsenseQA":
		sim_dataset = ClosedEndedExplanationDataset("CommonsenseQA", llm, gpt_sim=True)
	elif dataset_name == "WinoGrande":
		sim_dataset = ClosedEndedExplanationDataset(llm, gpt_sim=True)	
	elif dataset_name == "squad":
		sim_dataset = SquadExplanationDataset(llm, gpt_sim=True)
	elif dataset_name == "nq":
		sim_dataset = OpenEndedExplanationDataset(llm, gpt_sim=True)
	else:
		print("Other dataset")
		import sys
		sys.exit()
	
	b = True

	train_data, train_labels, train_log_probs = \
		dataset.train_data, dataset.train_labels, dataset.train_log_probs
	
	test_data, test_labels, test_log_probs, = \
		dataset.test_data, dataset.test_labels, dataset.test_log_probs
	
	train_logits, train_pre_conf, train_post_conf = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
	test_logits, test_pre_conf, test_post_conf = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

	# reshape pre_conf, post_conf, log_probs
	train_pre_conf = train_pre_conf.reshape(len(train_labels), -1)
	test_pre_conf = test_pre_conf.reshape(len(test_labels), -1)
	train_post_conf = train_post_conf.reshape(len(train_labels), -1)
	test_post_conf = test_post_conf.reshape(len(test_labels), -1)

	train_log_probs = train_log_probs.reshape(len(train_data), -1)
	test_log_probs = test_log_probs.reshape(len(test_data), -1)


	train_random_data = random_dataset.train_data
	test_random_data = random_dataset.test_data

	train_sim_data = sim_dataset.train_data
	test_sim_data = sim_dataset.test_data

	print("train_random_data", train_random_data.shape)
	print("train_data", train_data.shape)

	seeds = range(20)
	num_prompt_list = range(2, 41, 2)

	train_data_all = train_data
	test_data_all = test_data

	train_random_data_all = train_random_data
	test_random_data_all = test_random_data

	train_sim_data_all = train_sim_data
	test_sim_data_all = test_sim_data

	train_concat_all = np.concatenate([
		train_data, train_log_probs, train_pre_conf, train_post_conf,
		train_random_data, train_sim_data
	], axis=1)

	test_concat_all = np.concatenate([
		test_data, test_log_probs, test_pre_conf, test_post_conf,
		test_random_data, test_sim_data
	], axis=1)

	print(train_data.shape, test_data.shape)

	results = {}
	random_results = {}
	sim_results = {}
	for s in num_prompt_list:
		results[s] = []
		random_results[s] = []
		sim_results[s] = []

	for seed in seeds:
		# set random seed
		np.random.seed(seed)
		torch.manual_seed(seed)
		
		# randomly shuffle a list of inds to select
		idxs = np.random.permutation(train_random_data.shape[1])

		for num_prompts in num_prompt_list:
			prompt_ids = idxs[:num_prompts]

			train_data_subset = train_data_all[:, prompt_ids]
			test_data_subset = test_data_all[:, prompt_ids]

			train_random_data_subset = train_random_data_all[:, prompt_ids]
			test_random_data_subset = test_random_data_all[:, prompt_ids]

			train_sim_data_subset = train_sim_data_all[:, prompt_ids]
			test_sim_data_subset = test_sim_data_all[:, prompt_ids]

			# train_concat_subset = train_concat_all[:, prompt_ids]
			# test_concat_subset = test_concat_all[:, prompt_ids]

			# train predictor
			acc, f1, ece, auroc = get_linear_results(train_data_subset, train_labels, test_data_subset, test_labels, seed=seed, balanced=b)
			results[num_prompts].append(auroc)

			# train random predictor
			acc, f1, ece, auroc = get_linear_results(train_random_data_subset, train_labels, test_random_data_subset, test_labels, seed=seed, balanced=b)
			random_results[num_prompts].append(auroc)

			# train sim predictor
			acc, f1, ece, auroc = get_linear_results(train_sim_data_subset, train_labels, test_sim_data_subset, test_labels, seed=seed, balanced=b)
			sim_results[num_prompts].append(auroc)

	# plot results
	import matplotlib.pyplot as plt
	plt.figure()

	# average over seeds and compute std
	means = []
	stds = []

	random_means = []
	random_stds = []

	sim_means = []
	sim_stds = []

	all_means = []
	all_stds = []

	for s in num_prompt_list:
		means.append(np.mean(results[s]))
		stds.append(np.std(results[s]) / np.sqrt(len(seeds)))

		random_means.append(np.mean(random_results[s]))
		random_stds.append(np.std(random_results[s]) / np.sqrt(len(seeds)))

		sim_means.append(np.mean(sim_results[s]))
		sim_stds.append(np.std(sim_results[s]) / np.sqrt(len(seeds)))


	# print(means, random_means, sim_means)

	print("og", np.round(means[:5], 4))
	print("diverse", np.round(random_means[:5], 4))
	print("redundant", np.round(sim_means[:5], 4))

	# plt.errorbar(num_prompt_list, means, yerr=stds)
	plt.plot(num_prompt_list, means, label="Elicitation Questions")
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2)
	plt.xlabel("Number of Elicitation Prompts", fontsize=24)

	plt.plot(num_prompt_list, random_means, label="Diverse Elicitation Questions")
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(random_means, random_stds)], [m + s for m, s in zip(random_means, random_stds)], alpha=0.2)

	plt.plot(num_prompt_list, sim_means, label="Similar Elicitation Questions")
	plt.fill_between(num_prompt_list, [m - s for m, s in zip(sim_means, sim_stds)], [m + s for m, s in zip(sim_means, sim_stds)], alpha=0.2)

	# set tick size
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)
	plt.ylabel("AUROC", fontsize=24)

	plt.legend()

	# tight layout
	plt.tight_layout()

	plt.savefig("figs/vary_num_prompts_div_" + dataset_name + "_" + llm + ".png")
	plt.savefig("figs/vary_num_prompts_div_" + dataset_name + "_" + llm + ".pdf")

if __name__ == "__main__":

	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default="BooIQ")
	parser.add_argument("--llm", type=str, default="llama-70b")
	args = parser.parse_args()

	# set random seed
	np.random.seed(0)
	torch.manual_seed(0)

	# vary_train_data("CommonsenseQA", "mistral-8x7b")
	# vary_number_prompts("CommonsenseQA", "mistral-8x7b")
	
	# for dataset in ["HaluEval", "ToxicEval", "BooIQ"]:
		# vary_number_prompts(dataset, "llama3-8b")
		# vary_number_prompts(dataset, "llama3-70b")

	
	# llm = "llama3-8b"
	llm = "llama3-70b"
	# vary_number_prompts_random("BooIQ", llm)
	# vary_number_prompts_random("HaluEval", llm)
	# vary_number_prompts_random("ToxicEval", llm)
	# vary_number_prompts_random("squad", llm)
	# vary_number_prompts_random("CommonsenseQA", llm)
	# vary_number_prompts_random("nq", llm)

	vary_number_prompts_diverse(args.dataset, args.llm)