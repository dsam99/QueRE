import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from src.quere_oai import BooIQExplanationDataset_OAI
from src.utils import train_linear_model, compute_ece
from src.llm import load_llm
import sys
import argparse
from tqdm import tqdm

from data.code_dataset_oai import AdversarialCodeDataset

def discern_adv_model(dataset_name, llm):

    print("Dataset: ", dataset_name)
    print("LLM: ", llm)

    if dataset_name == "BooIQ":

        if "gpt" in llm:
            dataset = BooIQExplanationDataset_OAI("BooIQ", llm, adv=False)
            dataset_adv = BooIQExplanationDataset_OAI("BooIQ", llm, adv=True)

        else:
            print("Not implemented for non GPT models")
            sys.exit()

    elif dataset_name == "ToxicEval":
        if "gpt" in llm:
            dataset = BooIQExplanationDataset_OAI("ToxicEval", llm, adv=False)
            dataset_adv = BooIQExplanationDataset_OAI("ToxicEval", llm, adv=True)
        else:
            print("Not implemented for non GPT models")
            sys.exit()

    elif dataset_name == "HaluEval":

        if "gpt" in llm:
            dataset = BooIQExplanationDataset_OAI("HaluEval", llm, adv=False)
            dataset_adv = BooIQExplanationDataset_OAI("HaluEval", llm, adv=True)
        else:
            print("Not implemented for non GPT models")
            sys.exit()
    
    elif dataset_name == "code":
        dataset = AdversarialCodeDataset(llm)
        dataset_adv = AdversarialCodeDataset(llm, adv=True)

    train_data1, train_labels1, train_log_probs1 = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs
    
    test_data1, test_labels1, test_log_probs1, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs
    
    train_logits1, train_pre_conf1, train_post_conf1 = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
    test_logits1, test_pre_conf1, test_post_conf1 = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

    train_pre_conf1 = train_pre_conf1.reshape(len(train_data1), -1)
    test_pre_conf1 = test_pre_conf1.reshape(len(test_data1), -1)

    train_post_conf1 = train_post_conf1.reshape(len(train_data1), -1)
    test_post_conf1 = test_post_conf1.reshape(len(test_data1), -1)

    train_data2, train_labels2, train_log_probs2 = \
        dataset_adv.train_data, dataset_adv.train_labels, dataset_adv.train_log_probs
    
    test_data2, test_labels2, test_log_probs2, = \
        dataset_adv.test_data, dataset_adv.test_labels, dataset_adv.test_log_probs
    
    train_logits2, train_pre_conf2, train_post_conf2 = dataset_adv.train_logits, dataset_adv.train_pre_confs, dataset_adv.train_post_confs
    test_logits2, test_pre_conf2, test_post_conf2 = dataset_adv.test_logits, dataset_adv.test_pre_confs, dataset_adv.test_post_confs

    train_pre_conf2 = train_pre_conf2.reshape(len(train_data2), -1)
    test_pre_conf2 = test_pre_conf2.reshape(len(test_data2), -1)

    train_post_conf2 = train_post_conf2.reshape(len(train_data2), -1)
    test_post_conf2 = test_post_conf2.reshape(len(test_data2), -1)

    # construct task of distinguishing between datasets
    train_data = np.concatenate([train_data1, train_data2], axis=0)
    train_labels = np.concatenate([np.zeros(len(train_data1)), np.ones(len(train_data2))], axis=0)
    test_data = np.concatenate([test_data1, test_data2], axis=0)
    test_labels = np.concatenate([np.zeros(len(test_data1)), np.ones(len(test_data2))], axis=0)

    train_log_probs = np.concatenate([train_log_probs1, train_log_probs2], axis=0)
    test_log_probs = np.concatenate([test_log_probs1, test_log_probs2], axis=0)

    train_logits = np.concatenate([train_logits1, train_logits2], axis=0)
    test_logits = np.concatenate([test_logits1, test_logits2], axis=0)

    train_pre_conf = np.concatenate([train_pre_conf1, train_pre_conf2], axis=0)
    test_pre_conf = np.concatenate([test_pre_conf1, test_pre_conf2], axis=0)

    train_post_conf = np.concatenate([train_post_conf1, train_post_conf2], axis=0)
    test_post_conf = np.concatenate([test_post_conf1, test_post_conf2], axis=0)

    # train_rep = np.concatenate([train_rep1, train_rep2], axis=0)
    # test_rep = np.concatenate([test_rep1, test_rep2], axis=0)

    # train a linear model to distinguish between the two datasets
    clf = train_linear_model(train_data, train_labels, test_data, test_labels)
    y_pred = clf.predict(test_data)
    acc = (test_labels == y_pred).mean()
    print("Explanation acc: ", acc)

    clf = train_linear_model(train_log_probs, train_labels, test_log_probs, test_labels)
    y_pred = clf.predict(test_log_probs)
    acc = (test_labels == y_pred).mean()
    print("Logprob acc: ", acc)

    clf = train_linear_model(train_logits, train_labels, test_logits, test_labels)
    y_pred = clf.predict(test_logits)
    acc = (test_labels == y_pred).mean()
    print("Logits acc: ", acc)

    clf = train_linear_model(train_pre_conf, train_labels, test_pre_conf, test_labels)
    y_pred = clf.predict(test_pre_conf)
    acc = (test_labels == y_pred).mean()
    print("Preconf acc: ", acc)

    clf = train_linear_model(train_post_conf, train_labels, test_post_conf, test_labels)
    y_pred = clf.predict(test_post_conf)
    acc = (test_labels == y_pred).mean()
    print("Postconf acc: ", acc)

    # get results for preconf
    clf = train_linear_model(train_pre_conf, train_labels, test_pre_conf, test_labels)
    y_pred = clf.predict(test_pre_conf)
    acc = (test_labels == y_pred).mean()
    f1 = f1_score(test_labels, y_pred)

    # get results for exp all
    train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf, train_logits], axis=1)
    test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf, test_logits], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    
    print("QueRE acc: ", acc)
    print("Original Model Accuracy", np.mean(test_labels1))
    print("Adv Model Accuracy", np.mean(test_labels2))



if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="BooIQ")
    parser.add_argument("--llm", type=str, default="gpt-3.5-turbo-0125")
    args = parser.parse_args()

    discern_adv_model(args.dataset_name, args.llm)
    # discern_adv_model("HaluEval", "gpt-3.5-turbo-0125")
    # discern_adv_model("BooIQ", "gpt-3.5-turbo-0125")

