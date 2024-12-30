import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.quere_oai import BooIQExplanationDataset_OAI
from src.utils import train_linear_model
import argparse


from data.code_dataset_oai import AdversarialCodeDataset as AdversarialCodeDataset_OAI
from data.code_dataset_open_source import AdversarialCodeDataset

def discern_adv_type(train_dataset_name, test_dataset_name, train_llm, test_llm):

    print("Training Dataset: ", train_dataset_name)
    print("Testing Dataset: ", test_dataset_name)
    print("Train LLM: ", train_llm)
    print("Test LLM: ", test_llm)

    # Load training dataset (BooIQ)
    if train_dataset_name == "BooIQ" or train_dataset_name == "HaluEval":
        train_dataset = BooIQExplanationDataset_OAI(train_dataset_name, train_llm, adv=False, load_quere=True)
        train_dataset_adv = BooIQExplanationDataset_OAI(train_dataset_name, train_llm, adv=True, load_quere=True)

    elif train_dataset_name == "code":
        # train_dataset = AdversarialCodeDataset_OAI(train_llm)
        # train_dataset_adv = AdversarialCodeDataset_OAI(train_llm, adv=True)
        # train_dataset_adv = AdversarialCodeDataset_OAI(train_llm, adv2=True)
        # train_dataset_adv = AdversarialCodeDataset_OAI(train_llm, adv2=True)
        
        train_dataset_adv = AdversarialCodeDataset_OAI(train_llm)
        train_dataset = AdversarialCodeDataset_OAI(train_llm, adv=True)

    else:
        raise ValueError(f"Unsupported training dataset: {train_dataset_name}")

    # Load testing dataset (code)
    if test_dataset_name == "code":
        test_dataset = AdversarialCodeDataset_OAI(test_llm)
        test_dataset_adv = AdversarialCodeDataset_OAI(test_llm, adv2=True)
        # test_dataset_adv = AdversarialCodeDataset_OAI(test_llm, syc=True)
    else:
        raise ValueError(f"Unsupported testing dataset: {test_dataset_name}")

    # Prepare training data
    train_data1, train_labels1, train_log_probs1 = \
        train_dataset.train_data, train_dataset.train_labels, train_dataset.train_log_probs

    train_logits1, train_pre_conf1, train_post_conf1 = train_dataset.train_logits, train_dataset.train_pre_confs, train_dataset.train_post_confs
    train_sorted_logits1 = train_dataset.train_sorted_logits

    train_data2, train_labels2, train_log_probs2 = \
        train_dataset_adv.train_data, train_dataset_adv.train_labels, train_dataset_adv.train_log_probs

    train_logits2, train_pre_conf2, train_post_conf2 = train_dataset_adv.train_logits, train_dataset_adv.train_pre_confs, train_dataset_adv.train_post_confs
    train_sorted_logits2 = train_dataset_adv.train_sorted_logits

    # train_data2 = train_data2[:, :train_data1.shape[1]]

    train_pre_conf1 = train_pre_conf1.reshape(len(train_data1), -1)
    train_pre_conf2 = train_pre_conf2.reshape(len(train_data2), -1)
    train_post_conf1 = train_post_conf1.reshape(len(train_data1), -1)
    train_post_conf2 = train_post_conf2.reshape(len(train_data2), -1)

    train_data = np.concatenate([train_data1, train_data2], axis=0)
    train_labels = np.concatenate([np.zeros(len(train_data1)), np.ones(len(train_data2))], axis=0)
    train_log_probs = np.concatenate([train_log_probs1, train_log_probs2], axis=0)
    train_logits = np.concatenate([train_logits1, train_logits2], axis=0)
    train_pre_conf = np.concatenate([train_pre_conf1, train_pre_conf2], axis=0)
    train_post_conf = np.concatenate([train_post_conf1, train_post_conf2], axis=0)
    train_sorted_logits = np.concatenate([train_sorted_logits1, train_sorted_logits2], axis=0)

    # Prepare testing data
    test_data1, test_labels1, test_log_probs1 = \
        test_dataset.test_data, test_dataset.test_labels, test_dataset.test_log_probs

    test_logits1, test_pre_conf1, test_post_conf1 = test_dataset.test_logits, test_dataset.test_pre_confs, test_dataset.test_post_confs
    test_sorted_logits1 = test_dataset.test_sorted_logits

    test_data2, test_labels2, test_log_probs2 = \
        test_dataset_adv.test_data, test_dataset_adv.test_labels, test_dataset_adv.test_log_probs

    test_logits2, test_pre_conf2, test_post_conf2 = test_dataset_adv.test_logits, test_dataset_adv.test_pre_confs, test_dataset_adv.test_post_confs
    test_sorted_logits2 = test_dataset_adv.test_sorted_logits

    test_pre_conf1 = test_pre_conf1.reshape(len(test_data1), -1)
    test_pre_conf2 = test_pre_conf2.reshape(len(test_data2), -1)
    test_post_conf1 = test_post_conf1.reshape(len(test_data1), -1)
    test_post_conf2 = test_post_conf2.reshape(len(test_data2), -1)

    test_data = np.concatenate([test_data1, test_data2], axis=0)
    test_labels = np.concatenate([np.zeros(len(test_data1)), np.ones(len(test_data2))], axis=0)
    test_log_probs = np.concatenate([test_log_probs1, test_log_probs2], axis=0)
    test_logits = np.concatenate([test_logits1, test_logits2], axis=0)
    test_pre_conf = np.concatenate([test_pre_conf1, test_pre_conf2], axis=0)
    test_post_conf = np.concatenate([test_post_conf1, test_post_conf2], axis=0)
    test_sorted_logits = np.concatenate([test_sorted_logits1, test_sorted_logits2], axis=0)

    # print shapes
    print("Train Data Shape: ", train_data.shape)
    print("Train Labels Shape: ", train_labels.shape)
    print("Train Log Probs Shape: ", train_log_probs.shape)
    print("Train Logits Shape: ", train_logits.shape)
    print("Train Pre Confidence Shape: ", train_pre_conf.shape)
    print("Train Post Confidence Shape: ", train_post_conf.shape)

    print("Test Data Shape: ", test_data.shape)
    print("Test Labels Shape: ", test_labels.shape)
    print("Test Log Probs Shape: ", test_log_probs.shape)
    print("Test Logits Shape: ", test_logits.shape)
    print("Test Pre Confidence Shape: ", test_pre_conf.shape)
    print("Test Post Confidence Shape: ", test_post_conf.shape)

    print("Running Experiments")

    clf = train_linear_model(train_logits, train_labels, test_logits, test_labels, C=10)
    y_pred = clf.predict(test_logits)
    acc = (test_labels == y_pred).mean()
    print("Logits Accuracy: ", acc)
    print("Logits Train Accuracy: ", (train_labels == clf.predict(train_logits)).mean())

    clf = train_linear_model(train_pre_conf, train_labels, test_pre_conf, test_labels, C=0.1)
    y_pred = clf.predict(test_pre_conf)
    acc = (test_labels == y_pred).mean()
    print("Pre-confidence Accuracy: ", acc)
    print("Pre-confidence Train Accuracy: ", (train_labels == clf.predict(train_pre_conf)).mean())

    clf = train_linear_model(train_post_conf, train_labels, test_post_conf, test_labels, C=0.1)
    y_pred = clf.predict(test_post_conf)
    acc = (test_labels == y_pred).mean()
    print("Post-confidence Accuracy: ", acc)
    print("Post-confidence Train Accuracy: ", (train_labels == clf.predict(train_post_conf)).mean())

    train_data_all = np.concatenate([train_data, train_pre_conf, train_post_conf, train_sorted_logits], axis=1)
    test_data_all = np.concatenate([test_data, test_pre_conf, test_post_conf, test_sorted_logits], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels, C=0.001)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    print("Quere Features Accuracy: ", acc)



def discern_adv_transfer(dataset_name, llm1, llm2):

    print("Dataset: ", dataset_name)
    print("LLM1: ", llm1)
    print("LLM2: ", llm2)

    if dataset_name == "code":
        
        if "gpt" in llm1:
            dataset_1 = AdversarialCodeDataset_OAI(llm1)
            dataset_adv_1 = AdversarialCodeDataset_OAI(llm1, adv=True)
        else:
            dataset_1 = AdversarialCodeDataset(llm1)
            dataset_adv_1 = AdversarialCodeDataset(llm1, adv=True)
        
        if "gpt" in llm2:
            # dataset_2 = AdversarialCodeDataset_OAI(llm2)
            # dataset_adv_2 = AdversarialCodeDataset_OAI(llm2, adv=True)

            dataset_adv_2 = AdversarialCodeDataset_OAI(llm2)
            dataset_2 = AdversarialCodeDataset_OAI(llm2, adv=True)
        else:
            dataset_2 = AdversarialCodeDataset(llm2)
            dataset_adv_2 = AdversarialCodeDataset(llm2, adv=True)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    train_data1, train_labels1, train_log_probs1 = \
        dataset_1.train_data, dataset_1.train_labels, dataset_1.train_log_probs
    
    test_data1, test_labels1, test_log_probs1, = \
        dataset_2.test_data, dataset_2.test_labels, dataset_2.test_log_probs
    
    train_logits1, train_pre_conf1, train_post_conf1 = dataset_1.train_logits, dataset_1.train_pre_confs, dataset_1.train_post_confs
    test_logits1, test_pre_conf1, test_post_conf1 = dataset_2.test_logits, dataset_2.test_pre_confs, dataset_2.test_post_confs

    train_pre_conf1 = train_pre_conf1.reshape(len(train_data1), -1)
    test_pre_conf1 = test_pre_conf1.reshape(len(test_data1), -1)

    train_post_conf1 = train_post_conf1.reshape(len(train_data1), -1)
    test_post_conf1 = test_post_conf1.reshape(len(test_data1), -1)

    train_data2, train_labels2, train_log_probs2 = \
        dataset_adv_1.train_data, dataset_adv_1.train_labels, dataset_adv_1.train_log_probs
    
    test_data2, test_labels2, test_log_probs2, = \
        dataset_adv_2.test_data, dataset_adv_2.test_labels, dataset_adv_2.test_log_probs
    
    train_logits2, train_pre_conf2, train_post_conf2 = dataset_adv_1.train_logits, dataset_adv_1.train_pre_confs, dataset_adv_1.train_post_confs
    test_logits2, test_pre_conf2, test_post_conf2 = dataset_adv_2.test_logits, dataset_adv_2.test_pre_confs, dataset_adv_2.test_post_confs

    train_pre_conf2 = train_pre_conf2.reshape(len(train_data2), -1)
    test_pre_conf2 = test_pre_conf2.reshape(len(test_data2), -1)

    train_post_conf2 = train_post_conf2.reshape(len(train_data2), -1)
    test_post_conf2 = test_post_conf2.reshape(len(test_data2), -1)

    test_sorted_logits1 = dataset_2.test_sorted_logits
    test_sorted_logits2 = dataset_adv_2.test_sorted_logits

    # construct task of distinguishing between datasets
    train_data = np.concatenate([train_data1, train_data2], axis=0)
    train_labels = np.concatenate([np.zeros(len(train_data1)), np.ones(len(train_data2))], axis=0)
    test_data = np.concatenate([test_data1, test_data2], axis=0)
    test_labels = np.concatenate([np.zeros(len(test_data1)), np.ones(len(test_data2))], axis=0)

    train_log_probs = np.concatenate([train_log_probs1, train_log_probs2], axis=0)
    test_log_probs = np.concatenate([test_log_probs1, test_log_probs2], axis=0)

    train_logits = np.concatenate([train_logits1, train_logits2], axis=0)
    test_logits = np.concatenate([test_logits1, test_logits2], axis=0)

    test_sorted_logits = np.concatenate([test_sorted_logits1, test_sorted_logits2], axis=0)

    train_pre_conf = np.concatenate([train_pre_conf1, train_pre_conf2], axis=0)
    test_pre_conf = np.concatenate([test_pre_conf1, test_pre_conf2], axis=0)

    train_post_conf = np.concatenate([train_post_conf1, train_post_conf2], axis=0)
    test_post_conf = np.concatenate([test_post_conf1, test_post_conf2], axis=0)

    # train a linear model to distinguish between the two datasets
    clf = train_linear_model(train_data, train_labels, test_data, test_labels)
    y_pred = clf.predict(test_data)
    acc = (test_labels == y_pred).mean()
    print("Explanation acc: ", acc)

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

    # try to add on logits -> convert train logits (from llama) into the topk sorted ones
    train_logits = np.argsort(train_logits, axis=1)
    train_logits = train_logits[:, -5:]
    train_logits = train_logits[:, ::-1]

    train_data_all = np.concatenate([train_data, train_logits], axis=1)
    test_data_all = np.concatenate([test_data, test_sorted_logits], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    
    print("QueRE acc: ", acc)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_name", type=str, default="HaluEval")
    parser.add_argument("--test_dataset_name", type=str, default="code")
    parser.add_argument("--train_llm", type=str, default="gpt-4o-mini")
    parser.add_argument("--test_llm", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    # discern_adv_type(args.train_dataset_name, args.test_dataset_name, args.train_llm, args.test_llm)
    # discern_adv_type("code", "code", "gpt-4o-mini", "gpt-4o-mini")
    discern_adv_type("code", "code", "gpt-4o-mini", "gpt-4o-mini")

    # discern_adv_transfer("code", "gpt-3.5-turbo-0125", "gpt-4o-mini")
    # discern_adv_transfer("code", "gpt-4o-mini", "gpt-3.5-turbo-0125")
    # discern_adv_transfer("code", "gpt-4o-mini", "llama-7b")
    # discern_adv_transfer("code", "gpt-4o-mini", "llama-7b")
    # discern_adv_transfer("code", "llama-7b", "gpt-4o-mini")
