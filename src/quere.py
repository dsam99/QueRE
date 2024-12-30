import torch
import numpy as np
from tqdm import tqdm
import os

from src.llm import load_llm, get_left_pad, get_add_token
from data.dataset import BooIQDataset, CommonsenseQADataset, WinoGrandeDataset, NQOpenDataset, HaluEvalDataset, HateSpeechDataset, SquadDataset
from src.utils import (
    gpt_explanation_prompts, explanation_prompts, 
    random_prompts, gpt_diverse_explanation_prompts, gpt_similar_explanation_prompts,
    get_preconf_prompt, get_postconf_prompt, random_tokens_prompts
)
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seeds
np.random.seed(0)
torch.manual_seed(0)

class ClosedEndedExplanationDataset(torch.utils.data.Dataset):
    '''
    Datasets of elicited explanations for closed-ended question answering tasks
    Data: BooIQ, HaluEval, ToxicEval, CommonsenseQA

    Args:
    dataset_string (str): name of dataset
    model_type (str): name of LLM to extract features from
    random (bool): whether to use random sequences of language rather than elicitation questions
    gpt_exp (bool): whether to use the full set of QueRE prompts
    gpt_diverse (bool): whether to use a more diverse set of prompts
    gpt_sim (bool): whether to use a more similar/redundant set of prompts
    load_quere (bool): at loading time, this flag loads and appends the standard questions and the gpt_exp questions in QueRE
    '''

    def __init__(self, dataset_string, model_type, random=False, gpt_exp=False, gpt_diverse=False, gpt_sim=False,
                 load_quere=False, random_tokens=False):

        self.dataset_string = dataset_string        
        self.model = None

        # check if path exists
        folder_path = "./data/quere_datasets/" + dataset_string + "_outputs/" + model_type
        
        if gpt_exp: # standard elicitation questions
            folder_path += "_gpt"
        elif gpt_diverse: # diverse prompts
            folder_path += "_gpt_diverse"
        elif gpt_sim: # similar prompts
            folder_path += "_gpt_sim"
        elif random: # unrelated sequences of language
            folder_path += "_random"
        elif random_tokens: # random tokens
            folder_path += "_random_tokens"
        
        if "70b" in model_type:
            train_subset = 1000
            test_subset = 1000
        else:
            train_subset = 5000
            test_subset = 1000
        
        # check if folder path exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = folder_path + "/train_explanations.npy"
        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_explanations.npy"):

            print("No data found at " + path)
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if dataset_string == "BooIQ":
                self.train_dataset = BooIQDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = BooIQDataset(split="test", tokenizer=self.tokenizer)
                self.options = self.train_dataset.options # prompt options are ["False", "True"]

            elif dataset_string == "HaluEval":
                self.train_dataset = HaluEvalDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = HaluEvalDataset(split="test", tokenizer=self.tokenizer)
                self.options = self.train_dataset.options # prompt options are ("no", "yes")
            
            elif dataset_string == "ToxicEval":
                self.train_dataset = HateSpeechDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = HateSpeechDataset(split="test", tokenizer=self.tokenizer)
                self.options = self.train_dataset.options # prompt options are ("no", "yes")
            
            elif dataset_string == "CommonsenseQA":
                self.train_dataset = CommonsenseQADataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = CommonsenseQADataset(split="test", tokenizer=self.tokenizer)
                self.options = self.train_dataset.options # prompt options are ("A", ..., "E")

            elif dataset_string == "WinoGrande":
                self.train_dataset = WinoGrandeDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = WinoGrandeDataset(split="validation", tokenizer=self.tokenizer)
                self.options = self.train_dataset.options # prompt options are ("A", "B")

            else:
                raise ValueError("Dataset not found")

            self.train_dataset.questions = self.train_dataset.questions[:train_subset]
            self.train_dataset.answers = self.train_dataset.answers[:train_subset]
            self.test_dataset.questions = self.test_dataset.questions[:test_subset]
            self.test_dataset.answers = self.test_dataset.answers[:test_subset]
            
            self.model_type = model_type
            self.left_pad = get_left_pad(model_type)
            self.add_token = get_add_token(model_type)

            # resulting arrays
            self.train_data, self.test_data = [], []
            self.train_log_probs, self.test_log_probs = [], []
            self.train_pre_confs, self.test_pre_confs = [], []
            self.train_post_confs, self.test_post_confs = [], []
            self.train_logits, self.test_logits = [], []

            # current prompts to generate simple responses
            self.pre_conf_prompt = get_preconf_prompt()
            self.post_conf_prompt = get_postconf_prompt()

            if gpt_exp:
                self.explanation_prompts = gpt_explanation_prompts()
            elif gpt_diverse:
                self.explanation_prompts = gpt_diverse_explanation_prompts()
            elif gpt_sim:
                self.explanation_prompts = gpt_similar_explanation_prompts()
            elif random:
                self.explanation_prompts = random_prompts()
            elif random_tokens:
                self.explanation_prompts = random_tokens_prompts()
            else:
                self.explanation_prompts = explanation_prompts()

        if os.path.exists(path):
            self.train_data = np.load(folder_path + "/train_explanations.npy")
            self.train_labels = np.load(folder_path + "/train_labels.npy")
            self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
            self.train_pre_confs = np.load(folder_path + "/train_pre_confs.npy")
            self.train_post_confs = np.load(folder_path + "/train_post_confs.npy")
            self.train_logits = np.load(folder_path + "/train_logits.npy")

        else:
            self.train_data, self.train_log_probs, self.train_labels, self.train_pre_confs, \
                self.train_post_confs, self.train_logits = self.process_data("train")
            np.save(folder_path + "/train_explanations.npy", self.train_data)
            np.save(folder_path + "/train_labels.npy", self.train_labels)
            np.save(folder_path + "/train_log_probs.npy", self.train_log_probs)
            np.save(folder_path + "/train_pre_confs.npy", self.train_pre_confs)
            np.save(folder_path + "/train_post_confs.npy", self.train_post_confs)
            np.save(folder_path + "/train_logits.npy", self.train_logits)

        if os.path.exists(folder_path + "/test_explanations.npy"):
            self.test_data = np.load(folder_path + "/test_explanations.npy")
            self.test_labels = np.load(folder_path + "/test_labels.npy")
            self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
            self.test_pre_confs = np.load(folder_path + "/test_pre_confs.npy")
            self.test_post_confs = np.load(folder_path + "/test_post_confs.npy")
            self.test_logits = np.load(folder_path + "/test_logits.npy")

        else:
            self.test_data, self.test_log_probs, self.test_labels, self.test_pre_confs, \
                    self.test_post_confs, self.test_logits = self.process_data("test")
            np.save(folder_path + "/test_explanations.npy", self.test_data)
            np.save(folder_path + "/test_labels.npy", self.test_labels)
            np.save(folder_path + "/test_log_probs.npy", self.test_log_probs)
            np.save(folder_path + "/test_pre_confs.npy", self.test_pre_confs)
            np.save(folder_path + "/test_post_confs.npy", self.test_post_confs)
            np.save(folder_path + "/test_logits.npy", self.test_logits)
        
        if load_quere: # for inference time -> loading QueRE only
            og_folder_path = "./data/quere_datasets/" + dataset_string + "_outputs/" + model_type
            self.og_train_data = np.load(og_folder_path + "/train_explanations.npy")
            self.og_test_data = np.load(og_folder_path + "/test_explanations.npy")

            self.gpt_exp_train_data = np.load(og_folder_path + "_gpt/train_explanations.npy")
            self.gpt_exp_test_data = np.load(og_folder_path + "_gpt/test_explanations.npy")

            # set train data as concatenation
            self.train_data = np.concatenate((self.og_train_data, self.gpt_exp_train_data), axis=1)
            self.test_data = np.concatenate((self.og_test_data, self.gpt_exp_test_data), axis=1)

        # delete model
        if self.model is not None:
            del self.model
            gc.collect()

        # reshape log probs
        self.train_log_probs = self.train_log_probs.reshape(-1, 2)
        self.test_log_probs = self.test_log_probs.reshape(-1, 2)

        # convert labels from downstream task label to if model was correct
        model_preds = np.argmax(self.train_log_probs, axis=1)
        self.train_labels = (model_preds == self.train_labels).astype(int)

        model_preds = np.argmax(self.test_log_probs, axis=1)
        self.test_labels = (model_preds == self.test_labels).astype(int)

    def process_data(self, split):
            
        if split == "train":
            base_dataset = self.train_dataset
        else:
            base_dataset = self.test_dataset
        
        # only do 1000 examples for all_logits for memory
        base_questions = base_dataset.questions

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_logits = []
        all_labels = base_dataset.answers

        # get indices of yes and no tokens -> answers options for elicitation questions
        yes_token = "yes"
        no_token = "no"

        if self.add_token:
            yes_token_id = self.tokenizer.encode(yes_token, return_tensors="pt")[:, 1]
            no_token_id = self.tokenizer.encode(no_token, return_tensors="pt")[:, 1]
        else:
            yes_token_id = self.tokenizer.encode(yes_token, return_tensors="pt")
            no_token_id = self.tokenizer.encode(no_token, return_tensors="pt")
            yes_token_id = yes_token_id[:, 0]
            no_token_id = no_token_id[:, 0]

        # get answer ids in vocab
        answer_tokens = self.options
        if self.add_token:
            answer_token_ids = [self.tokenizer.encode(token, return_tensors="pt")[:, 1] for token in answer_tokens]
        else:
            answer_token_ids = [self.tokenizer.encode(token, return_tensors="pt") for token in answer_tokens]
            answer_token_ids = [token_id[:, 0] for token_id in answer_token_ids]

        # loop through dataset questions 
        for q_ind, q in tqdm(enumerate(base_questions), total=len(base_questions)):

            # get last token logits after question
            input_ids = self.tokenizer.encode(q, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()
            all_logits.append(logits)

            # get pre confidence score - append pre conf prompt to question
            input_ids = self.tokenizer.encode(q[:-7] + self.pre_conf_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()
            pre_conf_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
            pre_conf_dist = torch.nn.functional.softmax(pre_conf_dist, dim=0)
            pre_conf = pre_conf_dist[0].item()
            all_pre_confs.append(pre_conf)

            # get post confidence score - append post conf prompt after adding answer to question
            input_ids_y = self.tokenizer.encode(q + " " + yes_token + " " + self.post_conf_prompt, return_tensors="pt").to(device)
            input_ids_n = self.tokenizer.encode(q + " " + no_token + " " + self.post_conf_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits_y = self.model(input_ids_y, return_dict=True).logits[0, -1, :].cpu()
                logits_n = self.model(input_ids_n, return_dict=True).logits[0, -1, :].cpu()

            post_conf_dist_y = torch.stack([logits_y[yes_token_id], logits_y[no_token_id]], dim=0).squeeze()
            post_conf_dist_y = torch.nn.functional.softmax(post_conf_dist_y, dim=0)
            post_conf_y = post_conf_dist_y[0].item()
            post_conf_dist_n = torch.stack([logits_n[yes_token_id], logits_n[no_token_id]], dim=0).squeeze()
            post_conf_dist_n = torch.nn.functional.softmax(post_conf_dist_n, dim=0)
            post_conf_n = post_conf_dist_n[0].item()
            all_post_confs.append([post_conf_y, post_conf_n])

            # get distribution over answers
            input_prompt = q + " "
            input_ids = self.tokenizer.encode(input_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()
            
            prob_dist = torch.stack([logits[no_token_id], logits[yes_token_id]], dim=1)
            prob_dist = torch.nn.functional.softmax(prob_dist, dim=1)
            all_log_probs.append(prob_dist)

            # stack elicitation questions + answer options in parallel
            exp_input = [q + token + " " + exp for token in answer_tokens for exp in self.explanation_prompts]
            batch_size = 8
            exp_dist = np.zeros((len(exp_input),)) # store responses to explanation questions
            for i in range(0, len(exp_input), batch_size):
                
                tokenizer_input = exp_input[i:i+batch_size]
                tokenized = self.tokenizer(tokenizer_input, padding=True, return_tensors="pt", return_attention_mask=True)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)

                with torch.no_grad():
                    log_probs = self.model(input_ids, attention_mask=attention_mask, return_dict=True).logits
                    last_token_id = attention_mask.sum(1) - 1

                if not self.left_pad:
                    log_probs = log_probs[range(log_probs.shape[0]), last_token_id, :].squeeze()
                else:
                    log_probs = log_probs[:, -1, :].squeeze()
                
                log_probs_yes = log_probs[:, yes_token_id]
                log_probs_no = log_probs[:, no_token_id]
                dist = torch.stack([log_probs_yes, log_probs_no], axis=1).squeeze()
                exp_dist[i:i+batch_size] = torch.nn.functional.softmax(dist, dim=1)[:, 0].cpu().numpy()
            
            all_data.append(exp_dist)
    
            # del from memory
            del input_ids
            del logits
            gc.collect()

        all_data = np.array(all_data)
        all_log_probs = np.array(all_log_probs)
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_logits = np.array(all_logits)
        all_labels = np.array(all_labels)
        return all_data, all_log_probs, all_labels, all_pre_confs, all_post_confs, all_logits

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class OpenEndedExplanationDataset(torch.utils.data.Dataset):
    '''
    Datasets of elicited explanations for the Natural questions (open-ended question answering task)

    Args:
    model_type (str): name of LLM to extract features from
    random (bool): whether to use random sequences of language rather than elicitation questions
    gpt_exp (bool): whether to use the full set of QueRE prompts
    gpt_diverse (bool): whether to use a more diverse set of prompts
    gpt_sim (bool): whether to use a more similar/redundant set of prompts
    load_quere (bool): at loading time, this flag loads and appends the standard questions and the gpt_exp questions in QueRE
    '''
    def __init__(self, model_type, gpt_exp=False, random=False, load_quere=False, gpt_diverse=False, gpt_sim=False):
        
        self.model_type = model_type
        self.left_pad = get_left_pad(model_type)
        self.add_token = get_add_token(model_type)

        self.gpt_exp = gpt_exp
        self.random = random
        self.gpt_diverse = gpt_diverse
        self.gpt_sim = gpt_sim

        # check if path exists
        if gpt_exp:
            folder_path = "./data/quere_datasets/NQOpen_outputs/" + model_type + "_gpt"
        elif random:
            folder_path = "./data/quere_datasets/NQOpen_outputs/" + model_type + "_random"
        elif gpt_diverse:
            folder_path = "./data/quere_datasets/NQOpen_outputs/" + model_type + "_gpt_diverse"
        elif gpt_sim:
            folder_path = "./data/quere_datasets/NQOpen_outputs/" + model_type + "_gpt_sim"
        else:
            folder_path = "./data/quere_datasets/NQOpen_outputs/" + model_type
        
        path = folder_path + "/train_explanations.npy"

        if "70b" in model_type:
            self.train_subset = 1000
            self.test_subset = 1000
        else:
            self.train_subset = 5000
            self.test_subset = 1000

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_explanations.npy"):
            print("No data found at " + path)
            
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
            self.train_dataset = NQOpenDataset(split="train", tokenizer=self.tokenizer) # load for in-context examples
            self.test_dataset = NQOpenDataset(split="validation", tokenizer=self.tokenizer)

            num_context = 2
            self.context_examples = ""

            for i in range(num_context):
                self.context_examples += self.train_dataset.questions[i] + " " + self.train_dataset.answers[i][0] + "\n"
            
            # don't double use context examples
            self.train_dataset.questions = self.train_dataset.questions[num_context:]
            self.train_dataset.answers = self.train_dataset.answers[num_context:]

            # subset train_data
            self.train_dataset.questions = self.train_dataset.questions[:self.train_subset]
            self.train_dataset.answers = self.train_dataset.answers[:self.train_subset]

            # subset test_data
            self.test_dataset.questions = self.test_dataset.questions[:self.test_subset]
            self.test_dataset.answers = self.test_dataset.answers[:self.test_subset]

            # current prompts to generate simple responses...
            self.pre_conf_prompt = get_preconf_prompt()
            self.post_conf_prompt = get_postconf_prompt()

            if gpt_exp:
                self.explanation_prompts = gpt_explanation_prompts()
            elif random:
                self.explanation_prompts = random_prompts()
            elif gpt_diverse:
                self.explanation_prompts = gpt_diverse_explanation_prompts()
            elif gpt_sim:
                self.explanation_prompts = gpt_similar_explanation_prompts()
            else:
                self.explanation_prompts = explanation_prompts()

            # check if path exists, otherwise make
            if not os.path.exists(folder_path):
                print("Making dir", folder_path)
                os.makedirs(folder_path)

            # stores result in self.data and self.labels
            self.train_data, self.test_data = [], [] # explanation answers
            self.train_labels, self.test_labels = [], [] # is output correct on certain question
            self.train_log_probs, self.test_log_probs = [], [] # model log probs
            self.train_pre_confs, self.train_post_confs = [], [] # pre and post confidences
            self.test_pre_confs, self.test_post_confs = [], [] # pre and post confidences
            self.train_logits, self.test_logits = [], [] # logits

            if os.path.exists(path):
                self.train_data = np.load(folder_path + "/train_explanations.npy")
                self.train_labels = np.load(folder_path + "/train_labels.npy")
                self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
                self.train_pre_confs = np.load(folder_path + "/train_pre_confs.npy")
                self.train_post_confs = np.load(folder_path + "/train_post_confs.npy")
                self.train_logits = np.load(folder_path + "/train_logits.npy")

            else:
                self.train_data, self.train_labels, self.train_log_probs, \
                    self.train_pre_confs, self.train_post_confs, self.train_logits = self.process_data("train")
            
                # save result
                np.save(folder_path + "/train_explanations.npy", self.train_data)
                np.save(folder_path + "/train_labels.npy", self.train_labels)
                np.save(folder_path + "/train_log_probs.npy", self.train_log_probs)
                np.save(folder_path + "/train_pre_confs.npy", self.train_pre_confs)
                np.save(folder_path + "/train_post_confs.npy", self.train_post_confs)
                np.save(folder_path + "/train_logits.npy", self.train_logits)

            
            if os.path.exists(folder_path + "/test_explanations.npy"):
                self.test_data = np.load(folder_path + "/test_explanations.npy")
                self.test_labels = np.load(folder_path + "/test_labels.npy")
                self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
                self.test_pre_confs = np.load(folder_path + "/test_pre_confs.npy")    
                self.test_post_confs = np.load(folder_path + "/test_post_confs.npy")
                self.test_logits = np.load(folder_path + "/test_logits.npy")

            else:
                self.test_data, self.test_labels, self.test_log_probs, \
                    self.test_pre_confs, self.test_post_confs, self.test_logits = self.process_data("test")
                
                np.save(folder_path + "/test_explanations.npy", self.test_data)
                np.save(folder_path + "/test_labels.npy", self.test_labels)
                np.save(folder_path + "/test_log_probs.npy", self.test_log_probs)
                np.save(folder_path + "/test_pre_confs.npy", self.test_pre_confs)
                np.save(folder_path + "/test_post_confs.npy", self.test_post_confs)
                np.save(folder_path + "/test_logits.npy", self.test_logits)
        
            # delete model
            if self.model is not None:
                del self.model
                gc.collect()

        else:
            print("Loading data")
            self.train_data = np.load(folder_path + "/train_explanations.npy")
            self.train_labels = np.load(folder_path + "/train_labels.npy")
            self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
            self.train_pre_confs = np.load(folder_path + "/train_pre_confs.npy")
            self.train_post_confs = np.load(folder_path + "/train_post_confs.npy")
            self.train_logits = np.load(folder_path + "/train_logits.npy")

            self.test_data = np.load(folder_path + "/test_explanations.npy")
            self.test_labels = np.load(folder_path + "/test_labels.npy")
            self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
            self.test_pre_confs = np.load(folder_path + "/test_pre_confs.npy")
            self.test_post_confs = np.load(folder_path + "/test_post_confs.npy")
            self.test_logits = np.load(folder_path + "/test_logits.npy")
        
        if load_quere:
            og_folder_path = "./data/quere_datasets/NQOpen_outputs/" + model_type
            # load og data
            self.og_train_data = np.load(og_folder_path + "/train_explanations.npy")
            self.og_test_data = np.load(og_folder_path + "/test_explanations.npy")

            self.gpt_exp_train_data = np.load(folder_path + "_gpt/train_explanations.npy")
            self.gpt_exp_test_data = np.load(folder_path + "_gpt/test_explanations.npy")

            # set train data as concatenation
            self.train_data = np.concatenate((self.og_train_data, self.gpt_exp_train_data), axis=1)
            self.test_data = np.concatenate((self.og_test_data, self.gpt_exp_test_data), axis=1)
        
        if "70b" in model_type:
            self.train_data = self.train_data[:1000]
            self.train_labels = self.train_labels[:1000]
            self.train_log_probs = self.train_log_probs[:1000]
            self.train_pre_confs = self.train_pre_confs[:1000]
            self.train_post_confs = self.train_post_confs[:1000]
            self.train_logits = self.train_logits[:1000]

    def process_data(self, split):

        count = 0

        # get ids of yes and no token - used later
        yes_token = "yes"
        no_token = "no"
        if self.add_token:
            yes_token_id = self.tokenizer.encode(yes_token, return_tensors="pt")[:, 1]
            no_token_id = self.tokenizer.encode(no_token, return_tensors="pt")[:, 1]
        else:
            yes_token_id = self.tokenizer.encode(yes_token, return_tensors="pt")
            no_token_id = self.tokenizer.encode(no_token, return_tensors="pt")
            yes_token_id = yes_token_id[:, 0]
            no_token_id = no_token_id[:, 0]

        all_data = []
        all_labels = []
        model_log_probs = []
        pre_confs = []
        post_confs = []
        all_logits = []

        if split == "train":
            base_dataset = self.train_dataset
        else:
            base_dataset = self.test_dataset

        # loop through questions 
        for q_ind, q in tqdm(enumerate(base_dataset.questions), total=len(base_dataset.questions)):

            answers = base_dataset.answers[q_ind] # list of potential open ended answers
            answer_tokens = self.tokenizer(answers, padding=True, return_tensors="pt").input_ids.to(device)
            max_len = answer_tokens.shape[1]

            input_ids = self.tokenizer.encode(self.context_examples + q, return_tensors="pt").to(device)
            q_len = len(input_ids[0])
            
            # get highest probability generation from model
            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=q_len + max_len, num_return_sequences=1, do_sample=False)
                output = self.tokenizer.decode(output[0, len(input_ids[0]):], skip_special_tokens=True)
                output = output.strip()
            
            correct_flag = False # used to check if model did predict correctly
            if output not in answers:
                all_labels.append(0)
            else:
                all_labels.append(1)
            
            # get last layer logits
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits
                logits = logits[0, -1, :]
                all_logits.append(logits.cpu().numpy())

            # get pre confidence
            inputs = self.context_examples + q[:-7] + self.pre_conf_prompt
            with torch.no_grad():
                logits = self.model(self.tokenizer(inputs, return_tensors="pt").input_ids.to(device), return_dict=True).logits
                logits = logits[0, -1, :]
                pre_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
                pre_dist = torch.nn.functional.softmax(pre_dist, dim=0)
                pre_conf = pre_dist[0].cpu().numpy().flatten()
            pre_confs.append(pre_conf)

            # get post confidence from its generated answer
            inputs = self.context_examples + q + output + self.post_conf_prompt
            with torch.no_grad():
                logits = self.model(self.tokenizer(inputs, return_tensors="pt").input_ids.to(device), return_dict=True).logits
                logits = logits[0, -1, :]
                post_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
                post_dist = torch.nn.functional.softmax(post_dist, dim=0)
                post_conf = post_dist[0].cpu().numpy().flatten()
            post_confs.append(post_conf)

            # get model probabilities of generated answer
            inputs = self.context_examples + q + " " + output
            token_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")
            input_ids = token_dict.input_ids.to(device)
            
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits
                logits = logits[0]
                output_logits = logits[q_len - 1: -1, :] # getting dist shifted by one
                probabilities = torch.nn.functional.softmax(output_logits, dim=1)
                output_tokens = token_dict.input_ids[0, q_len:].cpu().numpy()

                log_probs = torch.log(probabilities[range(probabilities.shape[0]), output_tokens])
                log_probs = log_probs.sum().item()

            model_log_probs.append(log_probs)

            # del from memory
            del input_ids
            del logits
            gc.collect()
            
            # get explanation responses
            exp_inputs = [inputs + " " + exp for exp in self.explanation_prompts]
            token_dict = self.tokenizer(exp_inputs, padding=True, return_tensors="pt")
            input_ids = token_dict.input_ids.to(device)
            
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits
                last_token_id = token_dict.attention_mask.sum(1) - 1
            
            # get probability of yes (w.r.t. distribution [yes, no])
            if self.left_pad:
                logits = logits[:, -1, :]
            else:
                logits = logits[range(logits.shape[0]), last_token_id, :].squeeze()
            
            prob_dist = torch.stack([logits[:, yes_token_id], logits[:, no_token_id]], dim=1).squeeze()
            prob_dist = torch.nn.functional.softmax(prob_dist, dim=1)
            prob_dist = prob_dist[:, 0].cpu().numpy()
            prob_dist = prob_dist.reshape(-1, len(self.explanation_prompts))
            
            # store results
            all_data.append(prob_dist)
            del input_ids
            del logits
            gc.collect()

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.array(all_labels)
        model_log_probs = np.array(model_log_probs)
        pre_confs = np.array(pre_confs)
        post_confs = np.array(post_confs)
        all_logits = np.array(all_logits)

        return all_data, all_labels, model_log_probs, pre_confs, post_confs, all_logits
         
class SquadExplanationDataset(torch.utils.data.Dataset):
    '''
    Datasets of elicited explanations for the Squad dataset (open-ended question answering task)

    Args:
    model_type (str): name of LLM to extract features from
    random (bool): whether to use random sequences of language rather than elicitation questions
    gpt_exp (bool): whether to use the full set of QueRE prompts
    gpt_diverse (bool): whether to use a more diverse set of prompts
    gpt_sim (bool): whether to use a more similar/redundant set of prompts
    load_quere (bool): at loading time, this flag loads and appends the standard questions and the gpt_exp questions in QueRE
    '''    
    def __init__(self, model_type, gpt_exp=False, gpt_diverse=False, gpt_sim=False, random=False, load_quere=False, random_tokens=False):
        self.model_type = model_type
        self.left_pad = get_left_pad(model_type)
        self.add_token = get_add_token(model_type)

        if gpt_exp:
            folder_path = "./data/quere_datasets/squad_outputs/" + model_type + "_gpt_exp"
        elif gpt_diverse:
            folder_path = "./data/quere_datasets/squad_outputs/" + model_type + "_gpt_diverse"
        elif gpt_sim:
            folder_path = "./data/quere_datasets/squad_outputs/" + model_type + "_gpt_sim"
        elif random:
            folder_path = "./data/quere_datasets/squad_outputs/" + model_type + "_random"
        elif random_tokens:
            folder_path = "./data/quere_datasets/squad_outputs/" + model_type + "_random_tokens"
        else:
            folder_path = "./data/quere_datasets/squad_outputs/" + model_type
            
        path = folder_path + "/train_explanations.npy"

        if "70b" in model_type:
            self.train_subset = 1000
            self.test_subset = 1000
        else:
            self.train_subset = 5000
            self.test_subset = 1000

        # check if folder path exists
        if not os.path.exists(folder_path):
            print("Making dir", folder_path)
            os.makedirs(folder_path)

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_explanations.npy"):
            print("No data found at " + path)
            
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
            self.train_dataset = SquadDataset(split="train", tokenizer=self.tokenizer)
            self.test_dataset = SquadDataset(split="validation", tokenizer=self.tokenizer)

            self.train_dataset.questions = self.train_dataset.questions[:self.train_subset]
            self.train_dataset.answers = self.train_dataset.answers[:self.train_subset]

            self.test_dataset.questions = self.test_dataset.questions[:self.test_subset]
            self.test_dataset.answers = self.test_dataset.answers[:self.test_subset]

            # current prompts to generate simple responses...
            self.pre_conf_prompt = "Will you answer this question correctly? [/INST]"
            self.post_conf_prompt = "[INST] Did you answer this question correctly? [/INST]"

            if gpt_exp:
                self.explanation_prompts = gpt_explanation_prompts()
            elif random:
                self.explanation_prompts = random_prompts()
            elif gpt_diverse:
                self.explanation_prompts = gpt_diverse_explanation_prompts()
            elif gpt_sim:
                self.explanation_prompts = gpt_similar_explanation_prompts()
            elif random_tokens:
                self.explanation_prompts = random_tokens_prompts()
            else:
                self.explanation_prompts = explanation_prompts()

            # stores result in self.data and self.labels
            self.train_data, self.test_data = [], [] # explanation answers
            self.train_labels, self.test_labels = [], [] # is output correct on certain question
            self.train_log_probs, self.test_log_probs = [], [] # model log probs
            self.train_pre_confs, self.train_post_confs = [], [] # pre and post confidences
            self.test_pre_confs, self.test_post_confs = [], [] # pre and post confidences
            self.train_logits, self.test_logits = [], [] # logits

            if os.path.exists(path):
                self.train_data = np.load(folder_path + "/train_explanations.npy")
                self.train_labels = np.load(folder_path + "/train_labels.npy")
                self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
                self.train_pre_confs = np.load(folder_path + "/train_pre_confs.npy")
                self.train_post_confs = np.load(folder_path + "/train_post_confs.npy")
                self.train_logits = np.load(folder_path + "/train_logits.npy")

            else:
                self.train_data, self.train_labels, self.train_log_probs, \
                    self.train_pre_confs, self.train_post_confs, self.train_logits = self.process_data("train")
            
                # save result
                np.save(folder_path + "/train_explanations.npy", self.train_data)
                np.save(folder_path + "/train_labels.npy", self.train_labels)
                np.save(folder_path + "/train_log_probs.npy", self.train_log_probs)
                np.save(folder_path + "/train_pre_confs.npy", self.train_pre_confs)
                np.save(folder_path + "/train_post_confs.npy", self.train_post_confs)
                np.save(folder_path + "/train_logits.npy", self.train_logits)

            
            if os.path.exists(folder_path + "/test_explanations.npy"):
                self.test_data = np.load(folder_path + "/test_explanations.npy")
                self.test_labels = np.load(folder_path + "/test_labels.npy")
                self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
                self.test_pre_confs = np.load(folder_path + "/test_pre_confs.npy")    
                self.test_post_confs = np.load(folder_path + "/test_post_confs.npy")
                self.test_logits = np.load(folder_path + "/test_logits.npy")

            else:
                self.test_data, self.test_labels, self.test_log_probs, \
                    self.test_pre_confs, self.test_post_confs, self.test_logits = self.process_data("test")
                
                np.save(folder_path + "/test_explanations.npy", self.test_data)
                np.save(folder_path + "/test_labels.npy", self.test_labels)
                np.save(folder_path + "/test_log_probs.npy", self.test_log_probs)
                np.save(folder_path + "/test_pre_confs.npy", self.test_pre_confs)
                np.save(folder_path + "/test_post_confs.npy", self.test_post_confs)
                np.save(folder_path + "/test_logits.npy", self.test_logits)
        
            # delete model
            if self.model is not None:
                del self.model  
                gc.collect()

        else:
            print("Loading data")
            self.train_data = np.load(folder_path + "/train_explanations.npy")
            self.train_labels = np.load(folder_path + "/train_labels.npy")
            self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
            self.train_pre_confs = np.load(folder_path + "/train_pre_confs.npy")
            self.train_post_confs = np.load(folder_path + "/train_post_confs.npy")
            self.train_logits = np.load(folder_path + "/train_logits.npy")

            self.test_data = np.load(folder_path + "/test_explanations.npy")
            self.test_labels = np.load(folder_path + "/test_labels.npy")
            self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
            self.test_pre_confs = np.load(folder_path + "/test_pre_confs.npy")
            self.test_post_confs = np.load(folder_path + "/test_post_confs.npy")
            self.test_logits = np.load(folder_path + "/test_logits.npy")
        
        if load_quere:
            og_folder_path = "./data/quere_datasets/squad_outputs/" + model_type
            # load og data
            self.og_train_data = np.load(og_folder_path + "/train_explanations.npy")
            self.og_test_data = np.load(og_folder_path + "/test_explanations.npy")

            self.gpt_exp_train_data = np.load(folder_path + "_gpt_exp/train_explanations.npy")
            self.gpt_exp_test_data = np.load(folder_path + "_gpt_exp/test_explanations.npy")

            # check if they are the same size
            if self.og_train_data.shape[0] != self.gpt_exp_train_data.shape[0]:
                print("Different sizes of train data")
                print(self.og_train_data.shape, self.gpt_exp_train_data.shape)
                print(self.og_test_data.shape, self.gpt_exp_test_data.shape)

                # reshape gpt_exp data
                self.gpt_exp_train_data = self.gpt_exp_train_data.reshape(self.train_subset, -1)
                self.gpt_exp_test_data = self.gpt_exp_test_data.reshape(-1, self.gpt_exp_train_data.shape[1])
                
            # truncate gpt_exp_train_data if necessary
            self.gpt_exp_train_data = self.gpt_exp_train_data[:self.train_subset]
            self.og_train_data = self.og_train_data[:self.train_subset]
            # truncate gpt_exp_test_data if necessary
            
            self.gpt_exp_test_data = self.gpt_exp_test_data[:self.test_subset]
            self.og_test_data = self.og_test_data[:self.test_subset]

            print("reshaped data")
            print(self.og_train_data.shape, self.gpt_exp_train_data.shape)
            print(self.og_test_data.shape, self.gpt_exp_test_data.shape)

            # set train data as concatenation
            self.train_data = np.concatenate((self.og_train_data, self.gpt_exp_train_data), axis=1)
            self.test_data = np.concatenate((self.og_test_data, self.gpt_exp_test_data), axis=1)

        # make labels and others truncated to training length
        self.train_data = self.train_data[:self.train_subset]
        self.train_labels = self.train_labels[:self.train_subset]
        self.train_log_probs = self.train_log_probs[:self.train_subset]
        self.train_pre_confs = self.train_pre_confs[:self.train_subset]
        self.train_post_confs = self.train_post_confs[:self.train_subset]
        self.train_logits = self.train_logits[:self.train_subset]

    def process_data(self, split):

        # get ids of yes and no token - used later
        yes_token = "yes"
        no_token = "no"

        if self.add_token:
            yes_token_id = self.tokenizer.encode(yes_token, return_tensors="pt")[:, 1]
            no_token_id = self.tokenizer.encode(no_token, return_tensors="pt")[:, 1]
        else:
            yes_token_id = self.tokenizer.encode(yes_token, return_tensors="pt")
            no_token_id = self.tokenizer.encode(no_token, return_tensors="pt")
            yes_token_id = yes_token_id[:, 0]
            no_token_id = no_token_id[:, 0]

        all_data = []
        all_labels = []
        model_log_probs = []
        pre_confs = []
        post_confs = []
        all_logits = []

        if split == "train":
            base_dataset = self.train_dataset
        else:
            base_dataset = self.test_dataset

        # loop through questions 
        for q_ind, q in tqdm(enumerate(base_dataset.questions), total=len(base_dataset.questions)):

            answers = base_dataset.answers[q_ind] # answer subsequence
            answer_tokens = self.tokenizer(answers, return_tensors="pt").input_ids.to(device)
            max_len = answer_tokens.shape[1]

            input_ids = self.tokenizer.encode(q, return_tensors="pt").to(device)
            q_len = len(input_ids[0])

            # get highest probability generation from model
            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=q_len + max_len, num_return_sequences=1, do_sample=False)
                output = self.tokenizer.decode(output[0, len(input_ids[0]):], skip_special_tokens=True)
                output = output.strip()

            # check if output matches answer
            if answers.strip().lower() in output.strip().lower(): # handle case like "the" or added punctuation
                all_labels.append(1)
            else:
                all_labels.append(0)
            
            # get last layer logits
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits
                logits = logits[0, -1, :]
                all_logits.append(logits.cpu().numpy())

            # get pre confidence
            inputs = q[:-7] + self.pre_conf_prompt
            with torch.no_grad():
                logits = self.model(self.tokenizer(inputs, return_tensors="pt").input_ids.to(device), return_dict=True).logits
                logits = logits[0, -1, :]
                pre_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
                pre_dist = torch.nn.functional.softmax(pre_dist, dim=0)
                pre_conf = pre_dist[0].cpu().numpy().flatten()
            pre_confs.append(pre_conf)

            # get post confidence from its generated answer
            inputs = q + output + self.post_conf_prompt
            with torch.no_grad():
                logits = self.model(self.tokenizer(inputs, return_tensors="pt").input_ids.to(device), return_dict=True).logits
                logits = logits[0, -1, :]
                post_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
                post_dist = torch.nn.functional.softmax(post_dist, dim=0)
                post_conf = post_dist[0].cpu().numpy().flatten()
            post_confs.append(post_conf)

            # get model probabilities of generated answer
            inputs = q + " " + output
            token_dict = self.tokenizer(inputs, padding=True, return_tensors="pt")
            input_ids = token_dict.input_ids.to(device)
            
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits
                logits = logits[0]
                output_logits = logits[q_len - 1: -1, :] # getting dist shifted by one
                probabilities = torch.nn.functional.softmax(output_logits, dim=1)
                output_tokens = token_dict.input_ids[0, q_len:].cpu().numpy()

                log_probs = torch.log(probabilities[range(probabilities.shape[0]), output_tokens])
                log_probs = log_probs.sum().item()

            model_log_probs.append(log_probs)

            # del from memory
            del input_ids
            del logits
            gc.collect()
            
            # get explanation responses
            exp_inputs = [inputs + " " + exp for exp in self.explanation_prompts]
            batch_size = 24
            
            exp_dist = np.zeros((len(exp_inputs),)) # store responses to explanation questions
            for i in range(0, len(exp_inputs), batch_size):                
                tokenizer_input = exp_inputs[i:i+batch_size]
                tokenized = self.tokenizer(tokenizer_input, padding=True, return_tensors="pt", return_attention_mask=True)
                input_ids = tokenized.input_ids.to(device)
                attention_mask = tokenized.attention_mask.to(device)
            
                with torch.no_grad():
                    logits = self.model(input_ids, attention_mask=attention_mask, return_dict=True).logits
                    last_token_id = token_dict.attention_mask.sum(1) - 1
            
                # get probability of yes (w.r.t. distribution [yes, no])
                if self.left_pad:
                    logits = logits[:, -1, :]
                else:
                    logits = logits[range(logits.shape[0]), last_token_id, :].squeeze()
            
                prob_dist = torch.stack([logits[:, yes_token_id], logits[:, no_token_id]], dim=1).squeeze()
                prob_dist = torch.nn.functional.softmax(prob_dist, dim=1)
                prob_dist = prob_dist[:, 0].cpu().numpy()
                exp_dist[i:i+batch_size] = prob_dist

            all_data.append(exp_dist)

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        model_log_probs = np.array(model_log_probs)
        pre_confs = np.array(pre_confs)
        post_confs = np.array(post_confs)
        all_logits = np.array(all_logits)

        return all_data, all_labels, model_log_probs, pre_confs, post_confs, all_logits

if __name__ == "__main__":

    # test BoolExplanationDataset
    dataset = ClosedEndedExplanationDataset(base_dataset="BooIQ", model_type="llama7b", load_quere=True)
    print("Length of dataset: ", len(dataset))
    print("First item: ", dataset[0])