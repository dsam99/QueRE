import numpy as np
from sklearn.metrics import f1_score

from src.quere import ClosedEndedExplanationDataset, SquadExplanationDataset
from src.quere_oai import BooIQExplanationDataset_OAI, SquadExplanationDataset_OAI
from baselines.rep_dataset import RepDataset
from src.utils import train_linear_model

def discern_model_architecture(dataset_name, llm1, llm2):

    print("Dataset: ", dataset_name)
    print("LLM1: ", llm1)
    print("LLM2: ", llm2)

    if dataset_name == "squad":
        dataset1 = SquadExplanationDataset(llm1, load_quere=True)
        dataset2 = SquadExplanationDataset(llm2, load_quere=True)
    else:
        dataset1 = ClosedEndedExplanationDataset(dataset_name, llm1, load_quere=True)
        dataset2 = ClosedEndedExplanationDataset(dataset_name, llm2, load_quere=True)

    # get datasets from 1 and 2
    rep_dataset1 = RepDataset(dataset_name, llm1)
    train_rep1 = rep_dataset1.train_rep
    test_rep1 = rep_dataset1.test_rep

    rep_dataset2 = RepDataset(dataset_name, llm2)
    train_rep2 = rep_dataset2.train_rep
    test_rep2 = rep_dataset2.test_rep

    train_data1, train_labels1, train_log_probs1 = \
        dataset1.train_data, dataset1.train_labels, dataset1.train_log_probs
    
    test_data1, test_labels1, test_log_probs1, = \
        dataset1.test_data, dataset1.test_labels, dataset1.test_log_probs
    
    train_logits1, train_pre_conf1, train_post_conf1 = dataset1.train_logits, dataset1.train_pre_confs, dataset1.train_post_confs
    test_logits1, test_pre_conf1, test_post_conf1 = dataset1.test_logits, dataset1.test_pre_confs, dataset1.test_post_confs

    train_pre_conf1 = train_pre_conf1.reshape(len(train_labels1), -1)
    test_pre_conf1 = test_pre_conf1.reshape(len(test_labels1), -1)

    train_log_probs1 = train_log_probs1.reshape(len(train_labels1), -1)
    test_log_probs1 = test_log_probs1.reshape(len(test_labels1), -1)

    train_data2, train_labels2, train_log_probs2 = \
        dataset2.train_data, dataset2.train_labels, dataset2.train_log_probs
    
    test_data2, test_labels2, test_log_probs2, = \
        dataset2.test_data, dataset2.test_labels, dataset2.test_log_probs
    
    train_logits2, train_pre_conf2, train_post_conf2 = dataset2.train_logits, dataset2.train_pre_confs, dataset2.train_post_confs
    test_logits2, test_pre_conf2, test_post_conf2 = dataset2.test_logits, dataset2.test_pre_confs, dataset2.test_post_confs

    train_pre_conf2 = train_pre_conf2.reshape(len(train_labels2), -1)
    test_pre_conf2 = test_pre_conf2.reshape(len(test_labels2), -1)

    train_log_probs2 = train_log_probs2.reshape(len(train_labels2), -1)
    test_log_probs2 = test_log_probs2.reshape(len(test_labels2), -1)

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

    # clf = train_linear_model(train_rep, train_labels, test_rep, test_labels)
    # y_pred = clf.predict(test_rep)
    # acc = (test_labels == y_pred).mean()
    # print("Rep acc: ", acc)

    # get results for preconf
    clf = train_linear_model(train_pre_conf, train_labels, test_pre_conf, test_labels)
    y_pred = clf.predict(test_pre_conf)
    acc = (test_labels == y_pred).mean()
    f1 = f1_score(test_labels, y_pred)

    # get results for exp all
    train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf], axis=1)
    test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    print("Exp all acc: ", acc)

def discern_model_architecture_oai(dataset_name):

    if dataset_name == "squad":
        
        dataset1 = SquadExplanationDataset_OAI("gpt-3.5-turbo-0125")
        dataset2 = SquadExplanationDataset_OAI("gpt-4o-mini")

    elif dataset_name == "BooIQ":

        dataset1 = BooIQExplanationDataset_OAI("BooIQ", "gpt-3.5-turbo-0125")
        dataset2 = BooIQExplanationDataset_OAI("BooIQ", "gpt-4o-mini")

    train_data1, train_labels1, train_log_probs1 = \
        dataset1.train_data, dataset1.train_labels, dataset1.train_log_probs
    
    test_data1, test_labels1, test_log_probs1, = \
        dataset1.test_data, dataset1.test_labels, dataset1.test_log_probs
    
    train_logits1, train_pre_conf1, train_post_conf1 = dataset1.train_logits, dataset1.train_pre_confs, dataset1.train_post_confs
    test_logits1, test_pre_conf1, test_post_conf1 = dataset1.test_logits, dataset1.test_pre_confs, dataset1.test_post_confs

    train_sorted_logits1, test_sorted_logits1 = dataset1.train_sorted_logits, dataset1.test_sorted_logits

    train_pre_conf1 = train_pre_conf1.reshape(len(train_labels1), -1)
    test_pre_conf1 = test_pre_conf1.reshape(len(test_labels1), -1)

    train_post_conf1 = train_post_conf1.reshape(len(train_labels1), -1)
    test_post_conf1 = test_post_conf1.reshape(len(test_labels1), -1)

    train_log_probs1 = train_log_probs1.reshape(len(train_labels1), -1)
    test_log_probs1 = test_log_probs1.reshape(len(test_labels1), -1)

    train_data2, train_labels2, train_log_probs2 = \
        dataset2.train_data, dataset2.train_labels, dataset2.train_log_probs
    
    test_data2, test_labels2, test_log_probs2, = \
        dataset2.test_data, dataset2.test_labels, dataset2.test_log_probs

    train_logits2, train_pre_conf2, train_post_conf2 = dataset2.train_logits, dataset2.train_pre_confs, dataset2.train_post_confs
    test_logits2, test_pre_conf2, test_post_conf2 = dataset2.test_logits, dataset2.test_pre_confs, dataset2.test_post_confs

    train_sorted_logits2, test_sorted_logits2 = dataset2.train_sorted_logits, dataset2.test_sorted_logits

    train_pre_conf2 = train_pre_conf2.reshape(len(train_labels2), -1)
    test_pre_conf2 = test_pre_conf2.reshape(len(test_labels2), -1)

    train_post_conf2 = train_post_conf2.reshape(len(train_labels2), -1)
    test_post_conf2 = test_post_conf2.reshape(len(test_labels2), -1)

    train_log_probs2 = train_log_probs2.reshape(len(train_labels2), -1)
    test_log_probs2 = test_log_probs2.reshape(len(test_labels2), -1)

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

    train_sorted_logits = np.concatenate([train_sorted_logits1, train_sorted_logits2], axis=0)
    test_sorted_logits = np.concatenate([test_sorted_logits1, test_sorted_logits2], axis=0)

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
    train_data_all = np.concatenate([train_data, train_log_probs, train_pre_conf, train_post_conf, train_sorted_logits], axis=1)
    test_data_all = np.concatenate([test_data, test_log_probs, test_pre_conf, test_post_conf, test_sorted_logits], axis=1)

    clf = train_linear_model(train_data_all, train_labels, test_data_all, test_labels)
    y_pred = clf.predict(test_data_all)
    acc = (test_labels == y_pred).mean()
    print("Exp all acc: ", acc)


if __name__ == "__main__":

    # llm1 = "mistral-7b"
    # llm2 = "mistral-8x7b"
    
    # llm1 = "llama3-8b"
    # llm2 = "llama3-70b"
    llm1 = "llama3-3b"
    llm2 = "llama3-8b"
    # discern_model_architecture("BooIQ", llm1, llm2)
    # discern_model_architecture("HaluEval", llm1, llm2)
    # discern_model_architecture("squad", llm1, llm2)
    
    # discern_model_architecture_oai("BooIQ")
    discern_model_architecture_oai("squad")

