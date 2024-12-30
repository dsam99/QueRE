import torch
import numpy as np
from sklearn.metrics import f1_score

from src.quere_oai import BooIQExplanationDataset_OAI
from src.utils import train_linear_model, compute_ece
from src.llm import load_llm
import sys
import argparse
from tqdm import tqdm

def discern_cautious_model(dataset_name, llm):

    print("Dataset: ", dataset_name)
    print("LLM: ", llm)

    if dataset_name == "BooIQ":

        
        if "gpt" in llm:
            dataset = BooIQExplanationDataset_OAI("BooIQ", llm, cautious_system_prompt=False)
            dataset_cautious = BooIQExplanationDataset_OAI("BooIQ", llm, cautious_system_prompt=True)

        else:
            print("Not implemented for non GPT models")
            sys.exit()
    elif dataset_name == "ToxicEval":
        if "gpt" in llm:
            dataset = BooIQExplanationDataset_OAI("ToxicEval", llm, cautious_system_prompt=False)
            dataset_cautious = BooIQExplanationDataset_OAI("ToxicEval", llm, cautious_system_prompt=True)

        else:
            print("Not implemented for non GPT models")
            sys.exit()

    elif dataset_name == "HaluEval":

        if "gpt" in llm:
            dataset = BooIQExplanationDataset_OAI("HaluEval", llm, cautious_system_prompt=False)
            dataset_cautious = BooIQExplanationDataset_OAI("HaluEval", llm, cautious_system_prompt=True)
    
        else:
            print("Not implemented for non GPT models")
            sys.exit()

    train_data1, train_labels1, train_log_probs1 = \
        dataset.train_data, dataset.train_labels, dataset.train_log_probs
    
    test_data1, test_labels1, test_log_probs1, = \
        dataset.test_data, dataset.test_labels, dataset.test_log_probs
    
    train_logits1, train_pre_conf1, train_post_conf1 = dataset.train_logits, dataset.train_pre_confs, dataset.train_post_confs
    test_logits1, test_pre_conf1, test_post_conf1 = dataset.test_logits, dataset.test_pre_confs, dataset.test_post_confs

    train_pre_conf1 = train_pre_conf1.reshape(len(train_labels1), -1)
    test_pre_conf1 = test_pre_conf1.reshape(len(test_labels1), -1)

    train_post_conf1 = train_post_conf1.reshape(len(train_labels1), -1)
    test_post_conf1 = test_post_conf1.reshape(len(test_labels1), -1)

    train_data2, train_labels2, train_log_probs2 = \
        dataset_cautious.train_data, dataset_cautious.train_labels, dataset_cautious.train_log_probs
    
    test_data2, test_labels2, test_log_probs2, = \
        dataset_cautious.test_data, dataset_cautious.test_labels, dataset_cautious.test_log_probs
    
    train_logits2, train_pre_conf2, train_post_conf2 = dataset_cautious.train_logits, dataset_cautious.train_pre_confs, dataset_cautious.train_post_confs
    test_logits2, test_pre_conf2, test_post_conf2 = dataset_cautious.test_logits, dataset_cautious.test_pre_confs, dataset_cautious.test_post_confs

    train_pre_conf2 = train_pre_conf2.reshape(len(train_labels2), -1)
    test_pre_conf2 = test_pre_conf2.reshape(len(test_labels2), -1)

    train_post_conf2 = train_post_conf2.reshape(len(train_labels2), -1)
    test_post_conf2 = test_post_conf2.reshape(len(test_labels2), -1)

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
    print("Exp all acc: ", acc)

    print("Original Model Accuracy", np.mean(test_labels1))
    print("Adv Model Accuracy", np.mean(test_labels2))



def discern_between_models(dataset_name, llm1, llm2):

    print("Dataset: ", dataset_name)
    print("LLM1: ", llm1)
    print("LLM2: ", llm2)

    if dataset_name == "BooIQ":

        if "gpt" in llm1 and "gpt" in llm2:
            # Load uncautious data for training
            dataset_llm1_uncautious = BooIQExplanationDataset_OAI("BooIQ", llm1, cautious_system_prompt=False)
            dataset_llm2_uncautious = BooIQExplanationDataset_OAI("BooIQ", llm2, cautious_system_prompt=False)

            # Load cautious data for testing
            dataset_llm1_cautious = BooIQExplanationDataset_OAI("BooIQ", llm1, cautious_system_prompt=True)
            dataset_llm2_cautious = BooIQExplanationDataset_OAI("BooIQ", llm2, cautious_system_prompt=True)

        else:
            # Handle other LLMs if necessary
            pass  # You can add code here if needed

    # You can add other datasets similarly

    # Training data: combine uncautious data from both models
    train_data_llm1, train_labels_llm1, train_log_probs_llm1 = \
        dataset_llm1_uncautious.train_data, dataset_llm1_uncautious.train_labels, dataset_llm1_uncautious.train_log_probs

    train_data_llm2, train_labels_llm2, train_log_probs_llm2 = \
        dataset_llm2_uncautious.train_data, dataset_llm2_uncautious.train_labels, dataset_llm2_uncautious.train_log_probs

    train_logits_llm1, train_pre_conf_llm1, train_post_conf_llm1 = \
        dataset_llm1_uncautious.train_logits, dataset_llm1_uncautious.train_pre_confs, dataset_llm1_uncautious.train_post_confs

    train_logits_llm2, train_pre_conf_llm2, train_post_conf_llm2 = \
        dataset_llm2_uncautious.train_logits, dataset_llm2_uncautious.train_pre_confs, dataset_llm2_uncautious.train_post_confs

    # Reshape confidences
    train_pre_conf_llm1 = train_pre_conf_llm1.reshape(len(train_labels_llm1), -1)
    train_pre_conf_llm2 = train_pre_conf_llm2.reshape(len(train_labels_llm2), -1)

    train_post_conf_llm1 = train_post_conf_llm1.reshape(len(train_labels_llm1), -1)
    train_post_conf_llm2 = train_post_conf_llm2.reshape(len(train_labels_llm2), -1)

    # Testing data: combine cautious data from both models
    test_data_llm1, test_labels_llm1, test_log_probs_llm1 = \
        dataset_llm1_cautious.test_data, dataset_llm1_cautious.test_labels, dataset_llm1_cautious.test_log_probs

    test_data_llm2, test_labels_llm2, test_log_probs_llm2 = \
        dataset_llm2_cautious.test_data, dataset_llm2_cautious.test_labels, dataset_llm2_cautious.test_log_probs

    test_logits_llm1, test_pre_conf_llm1, test_post_conf_llm1 = \
        dataset_llm1_cautious.test_logits, dataset_llm1_cautious.test_pre_confs, dataset_llm1_cautious.test_post_confs

    test_logits_llm2, test_pre_conf_llm2, test_post_conf_llm2 = \
        dataset_llm2_cautious.test_logits, dataset_llm2_cautious.test_pre_confs, dataset_llm2_cautious.test_post_confs

    # Reshape confidences
    test_pre_conf_llm1 = test_pre_conf_llm1.reshape(len(test_labels_llm1), -1)
    test_pre_conf_llm2 = test_pre_conf_llm2.reshape(len(test_labels_llm2), -1)

    test_post_conf_llm1 = test_post_conf_llm1.reshape(len(test_labels_llm1), -1)
    test_post_conf_llm2 = test_post_conf_llm2.reshape(len(test_labels_llm2), -1)

    # Construct training data and labels
    train_data = np.concatenate([train_data_llm1, train_data_llm2], axis=0)
    train_labels = np.concatenate([np.zeros(len(train_data_llm1)), np.ones(len(train_data_llm2))], axis=0)  # Label 0 for llm1, 1 for llm2

    train_log_probs = np.concatenate([train_log_probs_llm1, train_log_probs_llm2], axis=0)
    train_logits = np.concatenate([train_logits_llm1, train_logits_llm2], axis=0)
    train_pre_conf = np.concatenate([train_pre_conf_llm1, train_pre_conf_llm2], axis=0)
    train_post_conf = np.concatenate([train_post_conf_llm1, train_post_conf_llm2], axis=0)

    # Construct testing data and labels
    test_data = np.concatenate([test_data_llm1, test_data_llm2], axis=0)
    test_labels = np.concatenate([np.zeros(len(test_data_llm1)), np.ones(len(test_data_llm2))], axis=0)  # Label 0 for llm1, 1 for llm2

    test_log_probs = np.concatenate([test_log_probs_llm1, test_log_probs_llm2], axis=0)
    test_logits = np.concatenate([test_logits_llm1, test_logits_llm2], axis=0)
    test_pre_conf = np.concatenate([test_pre_conf_llm1, test_pre_conf_llm2], axis=0)
    test_post_conf = np.concatenate([test_post_conf_llm1, test_post_conf_llm2], axis=0)

    # Now, train a classifier to distinguish between llm1 and llm2 using uncautious data
    # and evaluate on cautious data

    # Train a linear model
    clf = train_linear_model(train_data, train_labels, test_data, test_labels)
    y_pred = clf.predict(test_data)
    acc = (test_labels == y_pred).mean()
    print("Explanation acc: ", acc)

    # Repeat for other features
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

    # Get results for combined features
    train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf, train_logits], axis=1)
    test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf, test_logits], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    print("Combined features acc: ", acc)


def discern_harmful_helpful(dataset_name, llm):

    print("Dataset: ", dataset_name)
    print("LLM1: ", llm)

    if dataset_name == "BooIQ":

        # Load helpful data for training
        dataset_helpful1 = BooIQExplanationDataset_OAI("BooIQ", llm)
        dataset_helpful2 = BooIQExplanationDataset_OAI("BooIQ", llm, cautious_system_prompt=True)
        dataset_helpful3 = BooIQExplanationDataset_OAI("BooIQ", llm, cautious_system_prompt2=True)

        # Load harmful data for training
        dataset_harmful1 = BooIQExplanationDataset_OAI("BooIQ", llm, adv=True)
        dataset_harmful2 = BooIQExplanationDataset_OAI("BooIQ", llm, adv2=True)
        dataset_harmful3 = BooIQExplanationDataset_OAI("BooIQ", llm, adv3=True)
    
    elif dataset_name == "HaluEval":

        # Load helpful data for training
        dataset_helpful1 = BooIQExplanationDataset_OAI("HaluEval", llm)
        dataset_helpful2 = BooIQExplanationDataset_OAI("HaluEval", llm, cautious_system_prompt=True)
        dataset_helpful3 = BooIQExplanationDataset_OAI("HaluEval", llm, cautious_system_prompt2=True)

        # Load harmful data for training
        dataset_harmful1 = BooIQExplanationDataset_OAI("HaluEval", llm, adv=True)
        dataset_harmful2 = BooIQExplanationDataset_OAI("HaluEval", llm, adv2=True)
        dataset_harmful3 = BooIQExplanationDataset_OAI("HaluEval", llm, adv3=True)
    
    elif dataset_name == "ToxicEval":

        # Load helpful data for training
        dataset_helpful1 = BooIQExplanationDataset_OAI("ToxicEval", llm)
        dataset_helpful2 = BooIQExplanationDataset_OAI("ToxicEval", llm, cautious_system_prompt=True)
        dataset_helpful3 = BooIQExplanationDataset_OAI("ToxicEval", llm, cautious_system_prompt2=True)

        # Load harmful data for training
        dataset_harmful1 = BooIQExplanationDataset_OAI("ToxicEval", llm, adv=True)
        dataset_harmful2 = BooIQExplanationDataset_OAI("ToxicEval", llm, adv2=True)
        dataset_harmful3 = BooIQExplanationDataset_OAI("ToxicEval", llm, adv3=True)

    else:
        print("Not implemented for other datasets")
        sys.exit()

    helpful_train_data1, helpful_train_labels1, helpful_train_log_probs1 = \
        dataset_helpful1.train_data, dataset_helpful1.train_labels, dataset_helpful1.train_log_probs
    
    harmful_train_data1, harmful_train_labels1, harmful_train_log_probs1 = \
        dataset_harmful1.train_data, dataset_harmful1.train_labels, dataset_harmful1.train_log_probs
    
    helpful_train_data2, helpful_train_labels2, helpful_train_log_probs2 = \
        dataset_helpful2.train_data, dataset_helpful2.train_labels, dataset_helpful2.train_log_probs
    
    harmful_train_data2, harmful_train_labels2, harmful_train_log_probs2 = \
        dataset_harmful2.train_data, dataset_harmful2.train_labels, dataset_harmful2.train_log_probs
    
    helpful_train_data3, helpful_train_labels3, helpful_train_log_probs3 = \
        dataset_helpful3.train_data, dataset_helpful3.train_labels, dataset_helpful3.train_log_probs
    
    harmful_train_data3, harmful_train_labels3, harmful_train_log_probs3 = \
        dataset_harmful3.train_data, dataset_harmful3.train_labels, dataset_harmful3.train_log_probs
    
    helpful_train_logits1, helpful_train_pre_conf1, helpful_train_post_conf1 = \
        dataset_helpful1.train_logits, dataset_helpful1.train_pre_confs, dataset_helpful1.train_post_confs
    
    harmful_train_logits1, harmful_train_pre_conf1, harmful_train_post_conf1 = \
        dataset_harmful1.train_logits, dataset_harmful1.train_pre_confs, dataset_harmful1.train_post_confs
    
    helpful_train_logits2, helpful_train_pre_conf2, helpful_train_post_conf2 = \
        dataset_helpful2.train_logits, dataset_helpful2.train_pre_confs, dataset_helpful2.train_post_confs
    
    harmful_train_logits2, harmful_train_pre_conf2, harmful_train_post_conf2 = \
        dataset_harmful2.train_logits, dataset_harmful2.train_pre_confs, dataset_harmful2.train_post_confs
    
    helpful_train_logits3, helpful_train_pre_conf3, helpful_train_post_conf3 = \
        dataset_helpful3.train_logits, dataset_helpful3.train_pre_confs, dataset_helpful3.train_post_confs
    
    harmful_train_logits3, harmful_train_pre_conf3, harmful_train_post_conf3 = \
        dataset_harmful3.train_logits, dataset_harmful3.train_pre_confs, dataset_harmful3.train_post_confs
    
    helpful_train_sorted_logits1, helpful_test_sorted_logits1 = dataset_helpful1.train_sorted_logits, dataset_helpful1.test_sorted_logits
    helpful_train_sorted_logits2, helpful_test_sorted_logits2 = dataset_helpful2.train_sorted_logits, dataset_helpful2.test_sorted_logits
    helpful_train_sorted_logits3, helpful_test_sorted_logits3 = dataset_helpful3.train_sorted_logits, dataset_helpful3.test_sorted_logits

    harmful_train_sorted_logits1, harmful_test_sorted_logits1 = dataset_harmful1.train_sorted_logits, dataset_harmful1.test_sorted_logits
    harmful_train_sorted_logits2, harmful_test_sorted_logits2 = dataset_harmful2.train_sorted_logits, dataset_harmful2.test_sorted_logits
    harmful_train_sorted_logits3, harmful_test_sorted_logits3 = dataset_harmful3.train_sorted_logits, dataset_harmful3.test_sorted_logits

    # load tets data
    helpful_test_data1, helpful_test_labels1, helpful_test_log_probs1 = \
        dataset_helpful1.test_data, dataset_helpful1.test_labels, dataset_helpful1.test_log_probs
    
    harmful_test_data1, harmful_test_labels1, harmful_test_log_probs1 = \
        dataset_harmful1.test_data, dataset_harmful1.test_labels, dataset_harmful1.test_log_probs
    
    helpful_test_data2, helpful_test_labels2, helpful_test_log_probs2 = \
        dataset_helpful2.test_data, dataset_helpful2.test_labels, dataset_helpful2.test_log_probs
    
    harmful_test_data2, harmful_test_labels2, harmful_test_log_probs2 = \
        dataset_harmful2.test_data, dataset_harmful2.test_labels, dataset_harmful2.test_log_probs
    
    helpful_test_data3, helpful_test_labels3, helpful_test_log_probs3 = \
        dataset_helpful3.test_data, dataset_helpful3.test_labels, dataset_helpful3.test_log_probs
    
    harmful_test_data3, harmful_test_labels3, harmful_test_log_probs3 = \
        dataset_harmful3.test_data, dataset_harmful3.test_labels, dataset_harmful3.test_log_probs
    
    helpful_test_logits1, helpful_test_pre_conf1, helpful_test_post_conf1 = \
        dataset_helpful1.test_logits, dataset_helpful1.test_pre_confs, dataset_helpful1.test_post_confs
    
    harmful_test_logits1, harmful_test_pre_conf1, harmful_test_post_conf1 = \
        dataset_harmful1.test_logits, dataset_harmful1.test_pre_confs, dataset_harmful1.test_post_confs
    
    helpful_test_logits2, helpful_test_pre_conf2, helpful_test_post_conf2 = \
        dataset_helpful2.test_logits, dataset_helpful2.test_pre_confs, dataset_helpful2.test_post_confs
    
    harmful_test_logits2, harmful_test_pre_conf2, harmful_test_post_conf2 = \
        dataset_harmful2.test_logits, dataset_harmful2.test_pre_confs, dataset_harmful2.test_post_confs
    
    helpful_test_logits3, helpful_test_pre_conf3, helpful_test_post_conf3 = \
        dataset_helpful3.test_logits, dataset_helpful3.test_pre_confs, dataset_helpful3.test_post_confs
    
    harmful_test_logits3, harmful_test_pre_conf3, harmful_test_post_conf3 = \
        dataset_harmful3.test_logits, dataset_harmful3.test_pre_confs, dataset_harmful3.test_post_confs
    
    # Reshape confidences
    helpful_train_pre_conf1 = helpful_train_pre_conf1.reshape(len(helpful_train_labels1), -1)
    harmful_train_pre_conf1 = harmful_train_pre_conf1.reshape(len(harmful_train_labels1), -1)
    helpful_train_pre_conf2 = helpful_train_pre_conf2.reshape(len(helpful_train_labels2), -1)
    harmful_train_pre_conf2 = harmful_train_pre_conf2.reshape(len(harmful_train_labels2), -1)
    helpful_train_pre_conf3 = helpful_train_pre_conf3.reshape(len(helpful_train_labels3), -1)
    harmful_train_pre_conf3 = harmful_train_pre_conf3.reshape(len(harmful_train_labels3), -1)

    helpful_train_post_conf1 = helpful_train_post_conf1.reshape(len(helpful_train_labels1), -1)
    harmful_train_post_conf1 = harmful_train_post_conf1.reshape(len(harmful_train_labels1), -1)
    helpful_train_post_conf2 = helpful_train_post_conf2.reshape(len(helpful_train_labels2), -1)
    harmful_train_post_conf2 = harmful_train_post_conf2.reshape(len(harmful_train_labels2), -1)
    helpful_train_post_conf3 = helpful_train_post_conf3.reshape(len(helpful_train_labels3), -1)
    harmful_train_post_conf3 = harmful_train_post_conf3.reshape(len(harmful_train_labels3), -1) 

    helpful_test_pre_conf1 = helpful_test_pre_conf1.reshape(len(helpful_test_labels1), -1)
    harmful_test_pre_conf1 = harmful_test_pre_conf1.reshape(len(harmful_test_labels1), -1)
    helpful_test_pre_conf2 = helpful_test_pre_conf2.reshape(len(helpful_test_labels2), -1)
    harmful_test_pre_conf2 = harmful_test_pre_conf2.reshape(len(harmful_test_labels2), -1)
    helpful_test_pre_conf3 = helpful_test_pre_conf3.reshape(len(helpful_test_labels3), -1)
    harmful_test_pre_conf3 = harmful_test_pre_conf3.reshape(len(harmful_test_labels3), -1)

    helpful_test_post_conf1 = helpful_test_post_conf1.reshape(len(helpful_test_labels1), -1)
    harmful_test_post_conf1 = harmful_test_post_conf1.reshape(len(harmful_test_labels1), -1)
    helpful_test_post_conf2 = helpful_test_post_conf2.reshape(len(helpful_test_labels2), -1)
    harmful_test_post_conf2 = harmful_test_post_conf2.reshape(len(harmful_test_labels2), -1)
    helpful_test_post_conf3 = helpful_test_post_conf3.reshape(len(helpful_test_labels3), -1)
    harmful_test_post_conf3 = harmful_test_post_conf3.reshape(len(harmful_test_labels3), -1)

    # concatenate data
    train_data = np.concatenate([helpful_train_data1, harmful_train_data1, helpful_train_data2, harmful_train_data2, helpful_train_data3, harmful_train_data3], axis=0)
    train_labels = np.concatenate([np.zeros(len(helpful_train_data1)), np.ones(len(harmful_train_data1)), np.zeros(len(helpful_train_data2)), np.ones(len(harmful_train_data2)), np.zeros(len(helpful_train_data3)), np.ones(len(harmful_train_data3))], axis=0)

    train_log_probs = np.concatenate([helpful_train_log_probs1, harmful_train_log_probs1, helpful_train_log_probs2, harmful_train_log_probs2, helpful_train_log_probs3, harmful_train_log_probs3], axis=0)
    train_logits = np.concatenate([helpful_train_logits1, harmful_train_logits1, helpful_train_logits2, harmful_train_logits2, helpful_train_logits3, harmful_train_logits3], axis=0)
    train_pre_conf = np.concatenate([helpful_train_pre_conf1, harmful_train_pre_conf1, helpful_train_pre_conf2, harmful_train_pre_conf2, helpful_train_pre_conf3, harmful_train_pre_conf3], axis=0)
    train_post_conf = np.concatenate([helpful_train_post_conf1, harmful_train_post_conf1, helpful_train_post_conf2, harmful_train_post_conf2, helpful_train_post_conf3, harmful_train_post_conf3], axis=0)

    test_data = np.concatenate([helpful_test_data1, harmful_test_data1, helpful_test_data2, harmful_test_data2, helpful_test_data3, harmful_test_data3], axis=0)
    test_labels = np.concatenate([np.zeros(len(helpful_test_data1)), np.ones(len(harmful_test_data1)), np.zeros(len(helpful_test_data2)), np.ones(len(harmful_test_data2)), np.zeros(len(helpful_test_data3)), np.ones(len(harmful_test_data3))], axis=0)
                                 
    test_log_probs = np.concatenate([helpful_test_log_probs1, harmful_test_log_probs1, helpful_test_log_probs2, harmful_test_log_probs2, helpful_test_log_probs3, harmful_test_log_probs3], axis=0)
    test_logits = np.concatenate([helpful_test_logits1, harmful_test_logits1, helpful_test_logits2, harmful_test_logits2, helpful_test_logits3, harmful_test_logits3], axis=0)
    test_pre_conf = np.concatenate([helpful_test_pre_conf1, harmful_test_pre_conf1, helpful_test_pre_conf2, harmful_test_pre_conf2, helpful_test_pre_conf3, harmful_test_pre_conf3], axis=0)
    test_post_conf = np.concatenate([helpful_test_post_conf1, harmful_test_post_conf1, helpful_test_post_conf2, harmful_test_post_conf2, helpful_test_post_conf3, harmful_test_post_conf3], axis=0)

    train_sorted_logits = np.concatenate([helpful_train_sorted_logits1, harmful_train_sorted_logits1, helpful_train_sorted_logits2, harmful_train_sorted_logits2, helpful_train_sorted_logits3, harmful_train_sorted_logits3], axis=0)
    test_sorted_logits = np.concatenate([helpful_test_sorted_logits1, harmful_test_sorted_logits1, helpful_test_sorted_logits2, harmful_test_sorted_logits2, helpful_test_sorted_logits3, harmful_test_sorted_logits3], axis=0)
    
    # Train a linear model
    clf = train_linear_model(train_data, train_labels, test_data, test_labels)
    y_pred = clf.predict(test_data)
    acc = (test_labels == y_pred).mean()
    print("Explanation acc: ", acc)

    # Repeat for other features
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

    # Get results for combined features
    train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf, train_sorted_logits], axis=1)
    test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf, test_sorted_logits], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    print("QueRE acc: ", acc)

if __name__ == "__main__":

    # discern_cautious_model("BooIQ", "gpt-3.5-turbo-0125")
    # discern_between_models("BooIQ", "gpt-3.5-turbo-0125", "gpt-4o-mini")

    # discern_harmful_helpful("BooIQ", "gpt-3.5-turbo-0125")
    # discern_harmful_helpful("BooIQ", "gpt-4o-mini")
    # discern_harmful_helpful("HaluEval", "gpt-3.5-turbo-0125")
    # discern_harmful_helpful("HaluEval", "gpt-4o-mini")
    
    discern_harmful_helpful("ToxicEval", "gpt-3.5-turbo-0125")
    discern_harmful_helpful("ToxicEval", "gpt-4o-mini")