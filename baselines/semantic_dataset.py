import torch
import numpy as np
from tqdm import tqdm
import os

from src.llm import load_llm, get_paths_from_string, get_left_pad, get_add_token
from data.dataset import BooIQDataset, CommonsenseQADataset, WinoGrandeDataset, NQOpenDataset, HaluEvalDataset, HateSpeechDataset, SquadDataset
from src.utils import get_preconf_prompt, get_postconf_prompt
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seeds
np.random.seed(0)
torch.manual_seed(0)
 
class BoolSemanticDataset(torch.utils.data.Dataset):
    '''
    Datasets of semantic uncertainty extracted for boolean questions
    '''

    def __init__(self, base_dataset, model_type):

        self.base_dataset = base_dataset        
        self.model = None

        # check if path exists
        folder_path = "./data/quere_datasets/" + base_dataset + "_outputs/" + model_type
        
        if "70b" in model_type:
            train_subset = 1000
            test_subset = 1000
        else:
            train_subset = 5000
            test_subset = 1000
        
        # check if folder path exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = folder_path + "/train_semantic.npy"
        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_semantic.npy"):

            print("No data found at " + path)
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            if base_dataset == "BooIQ":
                self.train_dataset = BooIQDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = BooIQDataset(split="test", tokenizer=self.tokenizer)
                
            elif base_dataset == "HaluEval":
                self.train_dataset = HaluEvalDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = HaluEvalDataset(split="test", tokenizer=self.tokenizer)
            
            elif base_dataset == "ToxicEval":
                self.train_dataset = HateSpeechDataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = HateSpeechDataset(split="test", tokenizer=self.tokenizer)
                    
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
            self.train_post_confs, self.test_post_confs = [], []

            # current prompts to generate simple responses...
            self.pre_conf_prompt = get_preconf_prompt()
            self.post_conf_prompt = get_postconf_prompt()

        if os.path.exists(path):
            self.train_post_confs = np.load(folder_path + "/train_semantic.npy")
            self.train_labels = np.load(folder_path + "/train_sem_labels.npy")
            self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
            self.train_log_probs = self.train_log_probs[:, 0, :]

        else:
            self.train_labels, self.train_post_confs = self.process_data("train")
            np.save(folder_path + "/train_sem_labels.npy", self.train_labels)
            np.save(folder_path + "/train_semantic.npy", self.train_post_confs)

            self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
            self.train_log_probs = self.train_log_probs[:, 0, :]

        if os.path.exists(folder_path + "/test_semantic.npy"):
            self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
            self.test_labels = np.load(folder_path + "/test_sem_labels.npy")
            self.test_post_confs = np.load(folder_path + "/test_semantic.npy")
            self.test_log_probs = self.test_log_probs[:, 0, :]

        else:
            self.test_labels, self.test_post_confs = self.process_data("test")
            np.save(folder_path + "/test_sem_labels.npy", self.test_labels)
            np.save(folder_path + "/test_semantic.npy", self.test_post_confs)
            
            self.test_log_probs = np.load(folder_path + "/test_log_probs.npy")
            self.test_log_probs = self.test_log_probs[:, 0, :]

        # delete model
        if self.model is not None:
            del self.model
            gc.collect()

        print(self.train_log_probs.shape)
        model_preds = np.argmax(self.train_log_probs, axis=1)
        self.train_labels = (model_preds == self.train_labels).astype(int)

        model_preds = np.argmax(self.test_log_probs, axis=1)
        self.test_labels = (model_preds == self.test_labels).astype(int)

    def process_data(self, split):
            
        if split == "train":
            base_dataset = self.train_dataset
        else:
            base_dataset = self.test_dataset
        
        base_questions = base_dataset.questions
        
        all_post_confs = []
        all_labels = base_dataset.answers

        yes_tokens = ["yes", "true", "correct", "right", "yep"] # used for semantic similarity baseline
        no_tokens = ["no", "false", "incorrect", "wrong", "nope"]

        yes_token = "yes" # used for inputting as prompt
        no_token = "no" # used for inputting as prompt

        if self.add_token:
            yes_token_ids = self.tokenizer(yes_tokens, padding=True, return_tensors="pt")["input_ids"][:, 1]
            no_token_ids = self.tokenizer(no_tokens, padding=True, return_tensors="pt")["input_ids"][:, 1]
        else:
            yes_token_ids = self.tokenizer(yes_tokens, padding=True, return_tensors="pt")
            no_token_ids = self.tokenizer(no_tokens, padding=True, return_tensors="pt")
            yes_token_ids = yes_token_ids[:, 0]
            no_token_ids = no_token_ids[:, 0]

        # loop through questions 
        for q_ind, q in tqdm(enumerate(base_questions), total=len(base_questions)):

            # get post confidence score - append post conf prompt after adding answer to question
            input_ids_y = self.tokenizer.encode(q + " " + yes_token + " " + self.post_conf_prompt, return_tensors="pt").to(device)
            input_ids_n = self.tokenizer.encode(q + " " + no_token + " " + self.post_conf_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                logits_y = self.model(input_ids_y, return_dict=True).logits[0, -1, :].cpu()
                logits_n = self.model(input_ids_n, return_dict=True).logits[0, -1, :].cpu()

            dist_yes_y = logits_y[yes_token_ids].sum()
            dist_no_y = logits_y[no_token_ids].sum()

            dist_yes_n = logits_n[yes_token_ids].sum()
            dist_no_n = logits_n[no_token_ids].sum()

            post_conf_dist_y = torch.stack([dist_yes_y, dist_no_y], dim=0).squeeze()
            post_conf_dist_y = torch.nn.functional.softmax(post_conf_dist_y, dim=0)

            post_conf_dist_n = torch.stack([dist_yes_n, dist_no_n], dim=0).squeeze()
            post_conf_dist_n = torch.nn.functional.softmax(post_conf_dist_n, dim=0)

            post_conf_y = post_conf_dist_y[0].item()
            post_conf_n = post_conf_dist_n[0].item()
            all_post_confs.append([post_conf_y, post_conf_n])

        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        return all_labels, all_post_confs

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class WinoGrandeSemanticDataset(torch.utils.data.Dataset):
    '''
    Datasets of semantic uncertainty extracted for WinoGrande questions
    '''

    def __init__(self, model_type, split="train"):

        self.model = None

        self.model_type = model_type
        self.left_pad = get_left_pad(model_type)
        self.add_token = get_add_token(model_type)
        self.split = split

        
        if "70b" in model_type:
            train_subset = 1000
            test_subset = 1000
        else:
            train_subset = 5000
            test_subset = 1000

        # check if path exists
        folder_path = "./data/quere_datasets/WinoGrande_outputs/" + model_type

        # check if folder path exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if split == "train":
            path = folder_path + "/train_semantic.npy"
        else:
            path = folder_path + "/test_semantic.npy"

        if not os.path.exists(path):

            print("No data found at " + path)
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.train_dataset = WinoGrandeDataset(split="train", tokenizer=self.tokenizer)
            self.test_dataset = WinoGrandeDataset(split="validation", tokenizer=self.tokenizer)  # as test set has no label?

            self.train_dataset.questions = self.train_dataset.questions[:train_subset]
            self.train_dataset.options1 = self.train_dataset.options1[:train_subset]
            self.train_dataset.options2 = self.train_dataset.options2[:train_subset]
            self.test_dataset.questions = self.test_dataset.questions[:test_subset]
            self.test_dataset.options1 = self.test_dataset.options1[:test_subset]
            self.test_dataset.options2 = self.test_dataset.options2[:test_subset]

            self.train_dataset.answers = self.train_dataset.answers[:train_subset]
            self.test_dataset.answers = self.test_dataset.answers[:test_subset]

            self.pre_conf_prompt = get_preconf_prompt()
            self.post_conf_prompt = get_postconf_prompt()

            # process data
            self.labels, self.post_confs = self.process_data(split)
            if split == "train":
                np.save(folder_path + "/train_sem_labels.npy", self.labels)
                np.save(folder_path + "/train_semantic.npy", self.post_confs)
            else:
                np.save(folder_path + "/test_sem_labels.npy", self.labels)
                np.save(folder_path + "/test_semantic.npy", self.post_confs)

        else:
            if split == "train":
                self.post_confs = np.load(folder_path + "/train_semantic.npy")
                self.labels = np.load(folder_path + "/train_sem_labels.npy")
            else:
                self.post_confs = np.load(folder_path + "/test_semantic.npy")
                self.labels = np.load(folder_path + "/test_sem_labels.npy")

        # delete model
        if self.model is not None:
            del self.model
            gc.collect()

        if split == "train":
            self.log_probs = np.load(folder_path + "/train_log_probs.npy")
        else:
            self.log_probs = np.load(folder_path + "/test_log_probs.npy")

        # convert labels from downstream task label to if model was correct
        self.log_probs = self.log_probs.squeeze()
        model_preds = np.argmax(self.log_probs, axis=1)
        self.labels = (model_preds == self.labels).astype(int)

        # set data
        self.data = self.post_confs

    def process_data(self, split):

        if split == "train":
            base_dataset = self.train_dataset
        else:
            base_dataset = self.test_dataset

        base_questions = base_dataset.questions

        all_post_confs = []
        all_labels = base_dataset.answers

        option1_tokens = ["A", "a", "OptionA", "ChoiceA", "AnswerA", "First", "Initial", "Alpha"]
        option2_tokens = ["B", "b", "OptionB", "ChoiceB", "AnswerB", "Second", "Bravo"]

        if self.add_token:
            option1_token_ids = self.tokenizer(option1_tokens, padding=True, return_tensors="pt")["input_ids"][:, 1]
            option2_token_ids = self.tokenizer(option2_tokens, padding=True, return_tensors="pt")["input_ids"][:, 1]
        else:
            option1_token_ids = self.tokenizer(option1_tokens, padding=True, return_tensors="pt")
            option2_token_ids = self.tokenizer(option2_tokens, padding=True, return_tensors="pt")

            option1_token_ids = option1_token_ids[:, 0]
            option2_token_ids = option2_token_ids[:, 0]
        # loop through questions
        for q_ind, q in tqdm(enumerate(base_questions), total=len(base_questions)):

            # get post confidence scores for each option
            post_confs = []

            for option in [base_dataset.options1[q_ind], base_dataset.options2[q_ind]]:
                input_text = q + " " + option + " " + self.post_conf_prompt
                input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()

                dist_yes = logits[option1_token_ids].sum()
                dist_no = logits[option2_token_ids].sum()

                post_conf_dist = torch.stack([dist_yes, dist_no], dim=0).squeeze()
                post_conf_dist = torch.nn.functional.softmax(post_conf_dist, dim=0)

                post_conf = post_conf_dist[0].item()  # Probability of "yes" or equivalents
                post_confs.append(post_conf)

            all_post_confs.append(post_confs)

        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        return all_labels, all_post_confs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class MCQSemanticDataset(torch.utils.data.Dataset):
    '''
    Dataset of semantic uncertainty extracted for MCQ questions
    '''

    def __init__(self, base_dataset, model_type, split="train"):

        self.base_dataset = base_dataset
        self.model = None

        self.model_type = model_type
        self.left_pad = get_left_pad(model_type)
        self.add_token = get_add_token(model_type)
        self.split = split

        
        if "70b" in model_type:
            train_subset = 1000
            test_subset = 1000
        else:
            train_subset = 5000
            test_subset = 1000

        # Check if path exists
        folder_path = "./data/quere_datasets/" + self.base_dataset + "_outputs/" + model_type

        # Create folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if split == "train":
            path = folder_path + "/train_semantic.npy"
        else:
            path = folder_path + "/test_semantic.npy"

        if not os.path.exists(path):

            print("No data found at " + path)
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if base_dataset == "CommonsenseQA":
                self.train_dataset = CommonsenseQADataset(split="train", tokenizer=self.tokenizer)
                self.test_dataset = CommonsenseQADataset(split="test", tokenizer=self.tokenizer)
                self.options = self.train_dataset.options  # {'A': [...], 'B': [...], ...}
            else:
                raise ValueError("Dataset not found")

            # Select the appropriate dataset
            if split == "train":
                self.dataset = self.train_dataset
                self.dataset.questions = self.dataset.questions[:train_subset]
                self.dataset.answers = self.dataset.answers[:train_subset]
            else:
                self.dataset = self.test_dataset
                self.dataset.questions = self.dataset.questions[:test_subset]
                self.dataset.answers = self.dataset.answers[:test_subset]

            # Semantic equivalents for each option
            self.option_tokens = {
                'A': ["A", "a", "OptionA", "ChoiceA", "AnswerA", "First", "Initial", "Alpha"],
                'B': ["B", "b", "OptionB", "ChoiceB", "AnswerB", "Second", "Bravo"],
                'C': ["C", "c", "OptionC", "ChoiceC", "AnswerC", "Third", "Charlie"],
                'D': ["D", "d", "OptionD", "ChoiceD", "AnswerD", "Fourth", "Delta"],
                'E': ["E", "e", "OptionE", "ChoiceE", "AnswerE", "Fifth", "Echo"]
            }

            # Process data
            self.labels, self.option_probs = self.process_data(split)
            if split == "train":
                np.save(folder_path + "/train_sem_labels.npy", self.labels)
                np.save(folder_path + "/train_semantic.npy", self.option_probs)
            else:
                np.save(folder_path + "/test_sem_labels.npy", self.labels)
                np.save(folder_path + "/test_semantic.npy", self.option_probs)

        else:
            if split == "train":
                self.option_probs = np.load(folder_path + "/train_semantic.npy")
                self.labels = np.load(folder_path + "/train_sem_labels.npy")
            else:
                self.option_probs = np.load(folder_path + "/test_semantic.npy")
                self.labels = np.load(folder_path + "/test_sem_labels.npy")

        # Delete model to save memory
        if self.model is not None:
            del self.model
            gc.collect()

        # Load log_probs to compute model predictions
        if split == "train":
            self.log_probs = np.load(folder_path + "/train_log_probs.npy")
        else:
            self.log_probs = np.load(folder_path + "/test_log_probs.npy")

        # Convert labels from downstream task label to whether the model was correct
        # subset to 1000 if llama-70b
        if model_type == "llama-70b":
            self.log_probs = self.log_probs[:1000]
            self.labels = self.labels[:1000]
        self.log_probs = self.log_probs.squeeze()
        model_preds = np.argmax(self.log_probs, axis=1)
        self.labels = (model_preds == self.labels).astype(int)

        # Set data
        self.data = self.option_probs

    def process_data(self, split):

        base_dataset = self.dataset
        base_questions = base_dataset.questions
        all_option_probs = []
        all_labels = base_dataset.answers

        # Prepare option token IDs for semantic equivalents
        option_token_ids = {}
        for option, tokens in self.option_tokens.items():
            if self.add_token:
                token_ids = self.tokenizer(tokens, padding=True, return_tensors="pt")["input_ids"][:, 1]
            else:
                token_ids = self.tokenizer(tokens, padding=True, return_tensors="pt")["input_ids"][:, 0]
            option_token_ids[option] = token_ids

        # Loop through questions
        for q_ind, q in tqdm(enumerate(base_questions), total=len(base_questions)):

            # Encode the question
            input_ids = self.tokenizer.encode(q, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()

            # Convert to float if necessary
            if logits.dtype == torch.float16 or logits.dtype == torch.bfloat16:
                logits = logits.float()

            # Sum logits over semantic equivalents for each option
            option_logits = []
            for option_key in ['A', 'B', 'C', 'D', 'E']:
                token_ids = option_token_ids[option_key]
                option_logit = logits[token_ids].sum()
                option_logits.append(option_logit)

            # Compute probabilities over options
            option_logits = torch.stack(option_logits)
            option_probs = torch.nn.functional.softmax(option_logits, dim=0)
            all_option_probs.append(option_probs.numpy())

        all_option_probs = np.array(all_option_probs)
        all_labels = np.array(all_labels)

        # Map labels from 'A'-'E' to indices 0-4
        # option_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
        # all_labels = np.array([option_to_index[answer] for answer in all_labels])

        return all_labels, all_option_probs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]