import torch
import numpy as np
from tqdm import tqdm
import os
import sys
from src.utils import gpt_explanation_prompts, explanation_prompts, gpt_state_prompts
from data.dataset import BooIQDataset, CommonsenseQADataset, WinoGrandeDataset, NQOpenDataset, HaluEvalDataset, HateSpeechDataset, SquadDataset
from datasets import load_dataset
import pickle
from openai import OpenAI
from transformers import GPT2Tokenizer

valid_models = [
    "gpt-3.5-turbo-0125",
    "gpt-4-turbo-preview",
    "gpt-4o-mini",
]

ORG="<enter_org_here>"
PROJECT="<enter_project_here>"
KEY=os.environ.get("OPENAI_API_KEY_QUERE")

# set seeds
np.random.seed(0)
torch.manual_seed(0)

# use huggingface tokenizer for gpt2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def parse_response(response):
    '''
    Function to parse response from OpenAI API
    '''

    generation = response.choices[0].message.content
    logprobs = response.choices[0].logprobs.content[0].top_logprobs
    log_prob_dict = {d.token: d.logprob for d in logprobs}
    
    return generation, log_prob_dict

def get_yes_no_dist(log_prob_dict):
    '''
    Function to get distribution over "yes" and "no" tokens
    '''
    
    log_probs_yes = 0
    log_probs_no = 0

    for token, log_prob in log_prob_dict.items():
        token = token.lower()
        if "yes" in token:
            log_probs_yes += log_prob
        elif "no" in token:
            log_probs_no += log_prob
    
    # normalize 
    log_probs = np.array([log_probs_no, log_probs_yes])
    probs = np.exp(log_probs) / np.exp(log_probs).sum()
    return probs

def get_answer_dist(log_prob_dict, options):
    '''
    Function to get distribution over answer options
    '''
    
    log_probs = np.zeros(len(options))
    for token, log_prob in log_prob_dict.items():
        for i, option in enumerate(options):
            if option in token:
                log_probs[i] += log_prob
    
    # normalize 
    probs = np.exp(log_probs) / np.exp(log_probs).sum()
    return probs

class BooIQExplanationDataset_OAI(torch.utils.data.Dataset):

    def __init__(self, base_dataset, model_type, gpt_exp=False, gpt_state=False, adv=False, adv2=False, adv3=False,
                 cautious_system_prompt=False, cautious_system_prompt2=False, load_quere=False):
        
        self.base_dataset = base_dataset
        self.model_type = model_type
        self.adv = adv

        self.cautious_system_prompt = cautious_system_prompt
        self.cautious_system_prompt2 = cautious_system_prompt2
        self.adv2 = adv2
        self.adv3 = adv3

        if base_dataset == "BooIQ":
            self.options = ["no", "yes"]
        elif base_dataset == "ToxicEval":
            self.options = ["no", "yes"]
        elif base_dataset == "HaluEval":
            self.options = ["no", "yes"]

        if model_type not in valid_models:
            print("Invalid model type")
            sys.exit()

        if gpt_exp and adv:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp_adv"
        elif gpt_exp:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp"
        elif gpt_state:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_state"
        elif adv:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_adv"
        elif adv2:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_adv2"
        elif adv3:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_adv3"
        elif cautious_system_prompt:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_cautious_system_prompt"
        elif cautious_system_prompt2:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_cautious_system_prompt2"
        else:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type
        
        path = folder_path + "/train_responses.pkl"

        # current prompts to generate simple responses...
        self.pre_conf_prompt = "Will you answer this question correctly?"
        self.post_conf_prompt = "Did you answer this question correctly?"

        if gpt_exp:
            self.explanation_prompts = gpt_explanation_prompts()
        elif gpt_state:
            self.explanation_prompts = gpt_state_prompts()
        else:
            self.explanation_prompts = explanation_prompts()

        # check if folder path exists
        if not os.path.exists(folder_path):
            print("Making dir", folder_path)
            os.makedirs(folder_path)

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_responses.pkl"):
            print("No data found")

            if base_dataset == "BooIQ":
                train_dataset = BooIQDataset(split="train", template=False)
                test_dataset = BooIQDataset(split="test", template=False)
            elif base_dataset == "ToxicEval":
                train_dataset = HateSpeechDataset(split="train", template=False)
                test_dataset = HateSpeechDataset(split="test", template=False)
            elif base_dataset == "HaluEval":
                train_dataset = HaluEvalDataset(split="train", template=False)
                test_dataset = HaluEvalDataset(split="test", template=False)
            
            self.train_questions = train_dataset.questions
            self.train_answers = train_dataset.answers

            self.test_questions = test_dataset.questions
            self.test_answers = test_dataset.answers

            # subset train_data
            if "gpt-4-" in model_type:
                num_subset = 500
            else:
                # num_subset = 2000
                num_subset = 200
            self.train_questions = self.train_questions[:num_subset]
            self.train_answers = self.train_answers[:num_subset]
            
            # num_subset = 1000           
            num_subset = 100
            self.test_questions = self.test_questions[:num_subset]
            self.test_answers = self.test_answers[:num_subset]
            
            # remove [INST] and [\INST] from prompts bc not needed for openai models
            self.explanation_prompts = [exp.replace("[INST]", "").replace("[/INST]", "") for exp in self.explanation_prompts]

            if os.path.exists(path):
                train_responses = pickle.load(open(path, "rb"))
            
            else:
                if self.adv:
                    train_responses = self.process_data_sys_prompt("train", "You are an unhelpful assistant. Please answer questions incorrectly")
                elif self.cautious_system_prompt:
                    train_responses = self.process_data_sys_prompt("train", "You are a helpful and cautious assistant.")
                elif self.cautious_system_prompt2:
                    train_responses = self.process_data_sys_prompt("train", "You are a thoughtful chatbot who carefully considers questions and only provides solutions when the answers are clear so that we mitigate hallucinations.")
                elif self.adv2:
                    train_responses = self.process_data_sys_prompt("train", "You are an unhelpful and confident assistant. Please answer questions incorrectly.")
                elif self.adv3:
                    train_responses = self.process_data_sys_prompt("train", "You are a harmful AI system.")
                else:
                    train_responses = self.process_data("train")
                pickle.dump(train_responses, open(folder_path + "/train_responses.pkl", "wb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            if os.path.exists(folder_path + "/test_responses.pkl"):
                test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            else:
                if self.adv:
                    test_responses = self.process_data_sys_prompt("train", "You are an unhelpful assistant. Please answer questions incorrectly")
                elif self.cautious_system_prompt:
                    test_responses = self.process_data_sys_prompt("train", "You are a helpful and cautious assistant.")
                elif self.cautious_system_prompt2:
                    test_responses = self.process_data_sys_prompt("train", "You are a thoughtful chatbot who carefully considers questions and only provides solutions when the answers are clear so that we mitigate hallucinations.")
                elif self.adv2:
                    test_responses = self.process_data_sys_prompt("train", "You are an unhelpful and confident assistant. Please answer questions incorrectly.")
                elif self.adv3:
                    test_responses = self.process_data_sys_prompt("train", "You are a harmful AI system.")
                else:
                    test_responses = self.process_data("test")

                pickle.dump(test_responses, open(folder_path + "/test_responses.pkl", "wb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits= self.process_responses(test_responses)
            

        else:
            train_responses = pickle.load(open(path, "rb"))
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

        if load_quere: # load and append both base and gpt_exp data
            if adv:
                og_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_adv"
                added_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp_adv"

            else:
                og_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type
                added_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp"

            train_responses = pickle.load(open(og_path + "/train_responses.pkl", "rb"))
            train_responses_to_add = pickle.load(open(added_path + "/train_responses.pkl", "rb"))

            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            self.train_data_to_add, _, _, _, _, _, _ = self.process_responses(train_responses_to_add)

            print("Train data shape:", self.train_data.shape)
            print("Train data to add shape:", self.train_data_to_add.shape)

            self.train_data = np.concatenate([self.train_data, self.train_data_to_add], axis=1)

            test_responses = pickle.load(open(og_path + "/test_responses.pkl", "rb"))
            test_responses_to_add = pickle.load(open(added_path + "/test_responses.pkl", "rb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            
            self.test_data_to_add,_,_,_,_,_,_ = self.process_responses(test_responses_to_add)
            self.test_data = np.concatenate([self.test_data, self.test_data_to_add], axis=1)

    def process_data(self, split):

        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers


        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q + "Please answer with yes or no."},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "user", "content": q + "Please answer with yes or no."},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses
    
    def process_data_sys_prompt(self, split, system_prompt):
        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers


        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q + "Please answer with yes or no."},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses    
    
    def process_responses(self, responses):

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_labels = []
        sorted_logits = []
        all_logits = []

        for response in responses:
            generation = response["generation"]
            log_prob_dict = response["log_prob_dict"]
            answer = response["answer"]
            
            probs = get_answer_dist(log_prob_dict, self.options)
            all_log_probs.append(probs)

            # remove punctuation in generation
            generation = generation.replace("\"", "").replace("\'", "").replace(".", "")
            # print("Answer:", self.options[answer])
            # print("Generation:", generation)

            if self.options[answer] in generation.lower(): # check if answer is in generation
                all_labels.append(1)
            else:
                all_labels.append(0)

            # get sorted log prob values
            log_probs = np.array(list(log_prob_dict.values()))
            log_probs = np.sort(log_probs)[::-1]
            sorted_logits.append(log_probs)

            # map word in log_prob_dict to index in tokenizer vocab
            logits = np.zeros(len(tokenizer.get_vocab()))
            for token, log_prob in log_prob_dict.items():
                token_id = tokenizer.encode(token)[0]
                logits[token_id] = log_prob
            all_logits.append(logits)
                         
            # get pre confidence
            pre_log_prob_dict = response["pre_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(pre_log_prob_dict)
            all_pre_confs.append(probs[1])

            # get post confidence
            post_log_prob_dict = response["post_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(post_log_prob_dict)
            all_post_confs.append(probs[1])
            
            # get explanation responses
            exp_responses = np.zeros(len(self.explanation_prompts))
            # get explanation responses
            exp_list = []
            for k in response.keys():
                if "exp_log_prob_dict_" in k:
                    exp_list.append(k)

            exp_responses = np.zeros(len(exp_list))
            for ind, exp in enumerate(range(len(exp_list))):
                exp_log_prob_dict = response["exp_log_prob_dict_" + str(ind)]
                exp_probs = get_yes_no_dist(exp_log_prob_dict)
                exp_responses[ind] = exp_probs[1]
            all_data.append(exp_responses)

        all_data = np.array(all_data)
        all_log_probs = np.array(all_log_probs)
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        sorted_logits = np.array(sorted_logits)
        all_logits = np.array(all_logits)
        return all_data, all_labels, all_log_probs, all_pre_confs, all_post_confs, all_logits, sorted_logits

class MCQExplanationDataset_OAI(torch.utils.data.Dataset):
    
    def __init__(self, base_dataset, model_type, gpt_exp=False, gpt_state=False, load_quere=False):
        self.base_dataset = base_dataset
        self.model_type = model_type

        if base_dataset == "CommonsenseQA":
            self.options = ["A", "B", "C", "D", "E"]

        if model_type not in valid_models:
            print("Invalid model type")
            sys.exit()

        if gpt_exp:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp"
        elif gpt_state:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_state"
        else:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type
        
        path = folder_path + "/train_responses.pkl"

        # current prompts to generate simple responses...
        self.pre_conf_prompt = "Will you answer this question correctly?"
        self.post_conf_prompt = "Did you answer this question correctly?"

        if gpt_exp:
            self.explanation_prompts = gpt_explanation_prompts()
        elif gpt_state:
            self.explanation_prompts = gpt_state_prompts()
        else:
            self.explanation_prompts = explanation_prompts()

        # check if folder path exists
        if not os.path.exists(folder_path):
            print("Making dir", folder_path)
            os.makedirs(folder_path)

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_responses.pkl"):
            print("No data found")

            if base_dataset == "CommonsenseQA":
                train_dataset = CommonsenseQADataset(split="train", template=False)
                test_dataset = CommonsenseQADataset(split="test", template=False)
            
            self.train_questions = train_dataset.questions
            self.train_answers = train_dataset.answers

            self.test_questions = test_dataset.questions
            self.test_answers = test_dataset.answers

            # subset train_data
            if "gpt-4-" in model_type:
                num_subset = 500
            else:
                num_subset = 5000
            # num_subset = 5000
            self.train_questions = self.train_questions[:num_subset]
            self.train_answers = self.train_answers[:num_subset]
            
            # subset test_data
            num_subset = 1000           
            self.test_questions = self.test_questions[:num_subset]
            self.test_answers = self.test_answers[:num_subset]
            
            # remove [INST] and [\INST] from prompts bc not needed for openai models
            self.explanation_prompts = [exp.replace("[INST]", "").replace("[/INST]", "") for exp in self.explanation_prompts]

            if os.path.exists(path):
                train_responses = pickle.load(open(path, "rb"))
            
            else:
                train_responses = self.process_data("train")
                pickle.dump(train_responses, open(folder_path + "/train_responses.pkl", "wb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            if os.path.exists(folder_path + "/test_responses.pkl"):
                test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            else:
                test_responses = self.process_data("val")
                pickle.dump(test_responses, open(folder_path + "/test_responses.pkl", "wb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            

        else:
            train_responses = pickle.load(open(path, "rb"))
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

        if load_quere: # load and append both base and gpt_exp data
            og_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type
            added_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp"

            train_responses = pickle.load(open(og_path + "/train_responses.pkl", "rb"))
            train_responses_to_add = pickle.load(open(added_path + "/train_responses.pkl", "rb"))

            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            self.train_data_to_add, _, _, _, _, _, _ = self.process_responses(train_responses_to_add)

            print("Train data shape:", self.train_data.shape)
            print("Train data to add shape:", self.train_data_to_add.shape)

            self.train_data = np.concatenate([self.train_data, self.train_data_to_add], axis=1)

            test_responses = pickle.load(open(og_path + "/test_responses.pkl", "rb"))
            test_responses_to_add = pickle.load(open(added_path + "/test_responses.pkl", "rb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            
            self.test_data_to_add,_,_,_,_,_,_ = self.process_responses(test_responses_to_add)
            self.test_data = np.concatenate([self.test_data, self.test_data_to_add], axis=1)

            


    def process_data(self, split):

        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers


        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses
    def process_data_sys_prompt(self, split, system_prompt):
        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers

        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q + "Please answer with yes or no."},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses
        
    def process_responses(self, responses):

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_labels = []
        sorted_logits = []
        all_logits = []

        for response in responses:
            generation = response["generation"]
            log_prob_dict = response["log_prob_dict"]
            answer = response["answer"]
            
            probs = get_answer_dist(log_prob_dict, self.options)
            all_log_probs.append(probs)

            # get sorted log prob values
            log_probs = np.array(list(log_prob_dict.values()))
            log_probs = np.sort(log_probs)[::-1]
            sorted_logits.append(log_probs)

            # map word in log_prob_dict to index in tokenizer vocab
            logits = np.zeros(len(tokenizer.get_vocab()))
            for token, log_prob in log_prob_dict.items():
                token_id = tokenizer.encode(token)[0]
                logits[token_id] = log_prob
            all_logits.append(logits)

            # remove punctuation in generation
            generation = generation.replace("\"", "").replace("\'", "")
            if self.options[answer] in generation: # check if answer is in generation
                all_labels.append(1)
            else:
                all_labels.append(0)
            
            # get pre confidence
            pre_log_prob_dict = response["pre_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(pre_log_prob_dict)
            all_pre_confs.append(probs[1])

            # get post confidence
            post_log_prob_dict = response["post_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(post_log_prob_dict)
            all_post_confs.append(probs[1])
            
            # get explanation responses
            exp_responses = np.zeros(len(self.explanation_prompts))
            for ind, exp in enumerate(self.explanation_prompts):
                exp_log_prob_dict = response["exp_log_prob_dict_" + str(ind)]
                exp_probs = get_yes_no_dist(exp_log_prob_dict)
                exp_responses[ind] = exp_probs[1]
            all_data.append(exp_responses)

        all_data = np.array(all_data)
        all_log_probs = np.array(all_log_probs)
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        sorted_logits = np.array(sorted_logits)
        return all_data, all_labels, all_log_probs, all_pre_confs, all_post_confs, all_logits, sorted_logits


class WinoGrandeExplanationDataset_OAI(torch.utils.data.Dataset):
    
    def __init__(self, base_dataset, model_type, gpt_exp=False, adv=False,
                 load_quere = False):
        
        self.base_dataset = base_dataset
        self.model_type = model_type
        self.adv = adv

        self.options = ["A", "B"]
        if model_type not in valid_models:
            print("Invalid model type")
            sys.exit()

        if gpt_exp:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp"
        elif adv:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_adv"
        else:
            folder_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type
        path = folder_path + "/train_responses.pkl"

        # current prompts to generate simple responses...
        self.pre_conf_prompt = "Will you answer this question correctly?"
        self.post_conf_prompt = "Did you answer this question correctly?"

        if gpt_exp:
            self.explanation_prompts = gpt_explanation_prompts()
        else:
            self.explanation_prompts = explanation_prompts()

        # check if folder path exists
        if not os.path.exists(folder_path):
            print("Making dir", folder_path)
            os.makedirs(folder_path)

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_responses.pkl"):
            print("No data found")

            train_dataset = WinoGrandeDataset(split="train", template=False)
            test_dataset = WinoGrandeDataset(split="validation", template=False)
            
            self.train_questions = train_dataset.questions
            self.train_answers = train_dataset.answers

            self.test_questions = test_dataset.questions
            self.test_answers = test_dataset.answers

            # subset train_data
            if "gpt-4-" in model_type:
                num_subset = 500
            else:
                num_subset = 2000
            self.train_questions = self.train_questions[:num_subset]
            self.train_answers = self.train_answers[:num_subset]
            
            num_subset = 1000           
            self.test_questions = self.test_questions[:num_subset]
            self.test_answers = self.test_answers[:num_subset]
            
            # remove [INST] and [\INST] from prompts bc not needed for openai models
            self.explanation_prompts = [exp.replace("[INST]", "").replace("[/INST]", "") for exp in self.explanation_prompts]

            if os.path.exists(path):
                train_responses = pickle.load(open(path, "rb"))
            
            else:
                if self.adv:
                    train_responses = self.process_data_adv("train")
                else:
                    train_responses = self.process_data("train")
                pickle.dump(train_responses, open(folder_path + "/train_responses.pkl", "wb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            if os.path.exists(folder_path + "/test_responses.pkl"):
                test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            else:
                if self.adv:
                    test_responses = self.process_data_adv("test")
                else:
                    test_responses = self.process_data("test")
                pickle.dump(test_responses, open(folder_path + "/test_responses.pkl", "wb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            

        else:
            train_responses = pickle.load(open(path, "rb"))
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

        if load_quere: # load and append both base and gpt_exp data
            if adv:
                og_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_adv"
                added_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp_adv"

            else:
                og_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type
                added_path = "./data/quere_datasets/openai_exp/" + base_dataset + "_outputs/" + model_type + "_gpt_exp"

            train_responses = pickle.load(open(og_path + "/train_responses.pkl", "rb"))
            train_responses_to_add = pickle.load(open(added_path + "/train_responses.pkl", "rb"))

            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            self.train_data_to_add, _, _, _, _, _, _ = self.process_responses(train_responses_to_add)

            print("Train data shape:", self.train_data.shape)
            print("Train data to add shape:", self.train_data_to_add.shape)

            self.train_data = np.concatenate([self.train_data, self.train_data_to_add], axis=1)

            test_responses = pickle.load(open(og_path + "/test_responses.pkl", "rb"))
            test_responses_to_add = pickle.load(open(added_path + "/test_responses.pkl", "rb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            
            self.test_data_to_add,_,_,_,_,_,_ = self.process_responses(test_responses_to_add)
            self.test_data = np.concatenate([self.test_data, self.test_data_to_add], axis=1)


    def process_data(self, split):

        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers


        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q + "Please answer with yes or no."},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "user", "content": q + "Please answer with yes or no."},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses


    def process_data_sys_prompt(self, split, system_prompt):
        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers

        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q + "Please answer with yes or no."},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q + "Please answer with yes or no."},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses
    
    def process_responses(self, responses):

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_labels = []
        sorted_logits = []
        all_logits = []

        for response in responses:
            generation = response["generation"]
            log_prob_dict = response["log_prob_dict"]
            answer = response["answer"]
            
            probs = get_answer_dist(log_prob_dict, self.options)
            all_log_probs.append(probs)

            # remove punctuation in generation
            generation = generation.replace("\"", "").replace("\'", "").replace(".", "").strip()

            if self.options[answer] == generation[0]: # check if answer is in generation
                all_labels.append(1)
            else:
                all_labels.append(0)
            
            # get sorted log prob values
            log_probs = np.array(list(log_prob_dict.values()))
            log_probs = np.sort(log_probs)[::-1]
            sorted_logits.append(log_probs)

            # map word in log_prob_dict to index in tokenizer vocab
            logits = np.zeros(len(tokenizer.get_vocab()))
            for token, log_prob in log_prob_dict.items():
                token_id = tokenizer.encode(token)[0]
                logits[token_id] = log_prob
            all_logits.append(logits)
            
            # get pre confidence
            pre_log_prob_dict = response["pre_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(pre_log_prob_dict)
            all_pre_confs.append(probs[1])

            # get post confidence
            post_log_prob_dict = response["post_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(post_log_prob_dict)
            all_post_confs.append(probs[1])
            
            # get explanation responses
            exp_responses = np.zeros(len(self.explanation_prompts))
            for ind, exp in enumerate(self.explanation_prompts):
                exp_log_prob_dict = response["exp_log_prob_dict_" + str(ind)]
                exp_probs = get_yes_no_dist(exp_log_prob_dict)
                exp_responses[ind] = exp_probs[1]
            all_data.append(exp_responses)

        all_data = np.array(all_data)
        all_log_probs = np.array(all_log_probs)
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        sorted_logits = np.array(sorted_logits)
        return all_data, all_labels, all_log_probs, all_pre_confs, all_post_confs, all_logits, sorted_logits


class SquadExplanationDataset_OAI(torch.utils.data.Dataset):
    
    def __init__(self, model_type, gpt_exp=False, load_quere=False):
        self.model_type = model_type

        if model_type not in valid_models:
            print("Invalid model type")
            print(model_type)
            sys.exit()

        if gpt_exp:
            folder_path = "./data/quere_datasets/openai_exp/squad_outputs/" + model_type + "_gpt_exp"
        else:
            folder_path = "./data/quere_datasets/openai_exp/squad_outputs/" + model_type
        
        path = folder_path + "/train_responses.pkl"

        # current prompts to generate simple responses...
        self.pre_conf_prompt = "Will you answer this question correctly?"
        self.post_conf_prompt = "Did you answer this question correctly?"

        if gpt_exp:
            self.explanation_prompts = gpt_explanation_prompts()
        else:
            self.explanation_prompts = explanation_prompts()

        # check if folder path exists
        if not os.path.exists(folder_path):
            print("Making dir", folder_path)
            os.makedirs(folder_path)

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_responses.pkl"):
            print("No data found")
            
            train_dataset = load_dataset("squad")["train"]
            self.train_contexts = [q["context"] for q in train_dataset]
            self.train_questions = [q["question"] for q in train_dataset]
            self.train_answers = [q["answers"]["text"][0] for q in train_dataset]
            
            test_dataset = load_dataset("squad")["validation"]
            self.test_contexts = [q["context"] for q in test_dataset]
            self.test_questions = [q["question"] for q in test_dataset]
            self.test_answers = [q["answers"]["text"][0] for q in test_dataset]

            # subset train_data
            if "gpt-4-" in model_type:
                num_subset = 500
            else:
                num_subset = 5000
            # num_subset = 10
            self.train_contexts = self.train_contexts[:num_subset]
            self.train_questions = self.train_questions[:num_subset]
            self.train_answers = self.train_answers[:num_subset]
            
            # subset test_data
            num_subset = 1000 
            # num_subset = 5           
            self.test_contexts = self.test_contexts[:num_subset]
            self.test_questions = self.test_questions[:num_subset]
            self.test_answers = self.test_answers[:num_subset]
            
            # remove [INST] and [\INST] from prompts bc not needed for openai models
            self.explanation_prompts = [exp.replace("[INST]", "").replace("[/INST]", "") for exp in self.explanation_prompts]

            if os.path.exists(path):
                train_responses = pickle.load(open(path, "rb"))
            
            else:
                train_responses = self.process_data("train")
                pickle.dump(train_responses, open(folder_path + "/train_responses.pkl", "wb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            if os.path.exists(folder_path + "/test_responses.pkl"):
                test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            else:
                test_responses = self.process_data("val")
                pickle.dump(test_responses, open(folder_path + "/test_responses.pkl", "wb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            

        else:
            train_responses = pickle.load(open(path, "rb"))
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

        if load_quere: # load and append both base and gpt_exp data
            og_path = "./data/quere_datasets/openai_exp/squad_outputs/" + model_type
            added_path = "./data/quere_datasets/openai_exp/squad_outputs/" + model_type + "_gpt_exp"

            train_responses = pickle.load(open(og_path + "/train_responses.pkl", "rb"))
            train_responses_to_add = pickle.load(open(added_path + "/train_responses.pkl", "rb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            self.train_data_to_add, _, _, _, _, _, _ = self.process_responses(train_responses_to_add)
            
            print("Train data shape:", self.train_data.shape)
            print("Train data to add shape:", self.train_data_to_add.shape)

            self.train_data = np.concatenate([self.train_data, self.train_data_to_add], axis=1)

            test_responses = pickle.load(open(og_path + "/test_responses.pkl", "rb"))
            test_responses_to_add = pickle.load(open(added_path + "/test_responses.pkl", "rb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            
            self.test_data_to_add,_,_,_,_,_,_ = self.process_responses(test_responses_to_add)
            self.test_data = np.concatenate([self.test_data, self.test_data_to_add], axis=1)

    def process_data(self, split):

        all_responses = []

        if split == "train":
            contexts = self.train_contexts
            questions = self.train_questions
            answers = self.train_answers

        else:
            contexts = self.test_contexts
            questions = self.test_questions
            answers = self.test_answers

        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            context = contexts[q_ind]
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": "Please answer the following question using only the shortest sequence of text from the following context. Context: \"" + context + "\" \n \ Q: \"{" + q + "}\"?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": "Question: \"Please answer the following question using only the shortest sequence of text from the following context. Context: \"" + context + "\" \n \ Q: \"{" + q + "}\"? \""},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": "Question: \"Please answer the following question using only the shortest sequence of text from the following context. Context: \"" + context + "\" \n \ Q: \"{" + q + "}\"? \""},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "user", "content": "Question: \"Please answer the following question using only the shortest sequence of text from the following context. Context: \"" + context + "\" \n \ Q: \"{" + q + "}\"? \""},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses
    
    def process_responses(self, responses):

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_labels = []
        sorted_logits = []
        all_logits = []

        for response in responses:
            generation = response["generation"]
            log_prob_dict = response["log_prob_dict"]
            answer = response["answer"]
            
            probs = get_yes_no_dist(log_prob_dict)
            all_log_probs.append(probs[1])

            
            # remove punctuation in generation
            generation = generation.replace("\"", "").replace("\'", "")

            if answer.strip().lower() == generation.strip().lower(): # exact match only
                all_labels.append(1)
            else:
                all_labels.append(0)
            
            # print("Answer:", answer)
            # print("Generation:", generation)
            # print("label", all_labels[-1])

            # get sorted log prob values
            log_probs = np.array(list(log_prob_dict.values()))
            log_probs = np.sort(log_probs)[::-1]
            sorted_logits.append(log_probs)
            
            # map word in log_prob_dict to index in tokenizer vocab
            logits = np.zeros(len(tokenizer.get_vocab()))
            for token, log_prob in log_prob_dict.items():
                token_id = tokenizer.encode(token)[0]
                logits[token_id] = log_prob 
            all_logits.append(logits)

            # get pre confidence
            pre_log_prob_dict = response["pre_log_prob_dict"]


            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(pre_log_prob_dict)
            all_pre_confs.append(probs[1])

            # get post confidence
            post_log_prob_dict = response["post_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(post_log_prob_dict)
            all_post_confs.append(probs[1])
            
            # get explanation responses
            exp_list = []
            for k in response.keys():
                if "exp_log_prob_dict_" in k:
                    exp_list.append(k)

            exp_responses = np.zeros(len(exp_list))
            for ind, exp in enumerate(range(len(exp_list))):
                exp_log_prob_dict = response["exp_log_prob_dict_" + str(ind)]
                exp_probs = get_yes_no_dist(exp_log_prob_dict)
                exp_responses[ind] = exp_probs[1]
            all_data.append(exp_responses)

        all_data = np.array(all_data)
        all_log_probs = np.array(all_log_probs)
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        return all_data, all_labels, all_log_probs, all_pre_confs, all_post_confs, all_logits, sorted_logits


class OpenEndedExplanationDataset_OAI(torch.utils.data.Dataset):
    
    def __init__(self, model_type, gpt_exp=False, load_quere=False):
        self.model_type = model_type

        if model_type not in valid_models:
            print("Invalid model type")
            sys.exit()

        if gpt_exp:
            folder_path = "./data/quere_datasets/openai_exp/NQOpen_outputs/" + model_type + "_gpt_exp"
        else:
            folder_path = "./data/quere_datasets/openai_exp/NQOpen_outputs/" + model_type
        
        path = folder_path + "/train_responses.pkl"

        # current prompts to generate simple responses...
        self.pre_conf_prompt = "Will you answer this question correctly?"
        self.post_conf_prompt = "Did you answer this question correctly?"

        if gpt_exp:
            self.explanation_prompts = gpt_explanation_prompts()
        else:
            self.explanation_prompts = explanation_prompts()

        # check if folder path exists
        if not os.path.exists(folder_path):
            print("Making dir", folder_path)
            os.makedirs(folder_path)

        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_responses.pkl"):
            print("No data found")
            
            self.train_dataset = NQOpenDataset(split="train", template=False) # load for in-context examples
            self.test_dataset = NQOpenDataset(split="validation", template=False)
        
            self.train_questions = self.train_dataset.questions
            self.train_answers = self.train_dataset.answers

            self.test_questions = self.test_dataset.questions
            self.test_answers = self.test_dataset.answers

            # subset train_data
            if "gpt-4-" in model_type:
                num_subset = 500
            else:
                num_subset = 2000

            self.train_questions = self.train_questions[:num_subset]
            self.train_answers = self.train_answers[:num_subset]
            
            # subset test_data
            num_subset = 1000 
            self.test_questions = self.test_questions[:num_subset]
            self.test_answers = self.test_answers[:num_subset]
            
            # remove [INST] and [\INST] from prompts bc not needed for openai models
            self.explanation_prompts = [exp.replace("[INST]", "").replace("[/INST]", "") for exp in self.explanation_prompts]

            if os.path.exists(path):
                train_responses = pickle.load(open(path, "rb"))
            
            else:
                train_responses = self.process_data("train")
                pickle.dump(train_responses, open(folder_path + "/train_responses.pkl", "wb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            if os.path.exists(folder_path + "/test_responses.pkl"):
                test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            else:
                test_responses = self.process_data("val")
                pickle.dump(test_responses, open(folder_path + "/test_responses.pkl", "wb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            

        else:
            train_responses = pickle.load(open(path, "rb"))
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            test_responses = pickle.load(open(folder_path + "/test_responses.pkl", "rb"))
            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

        if load_quere: # load and append both base and gpt_exp data
            og_path = "./data/quere_datasets/openai_exp/NQOpen_outputs/" + model_type
            added_path = "./data/quere_datasets/openai_exp/NQOpen_outputs/" + model_type + "_gpt_exp"

            train_responses = pickle.load(open(og_path + "/train_responses.pkl", "rb"))
            train_responses_to_add = pickle.load(open(added_path + "/train_responses.pkl", "rb"))
            
            self.train_data, self.train_labels, self.train_log_probs, \
                self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)
            
            self.train_data_to_add, _, _, _, _, _, _ = self.process_responses(train_responses_to_add)
            
            print("Train data shape:", self.train_data.shape)
            print("Train data to add shape:", self.train_data_to_add.shape)

            self.train_data = np.concatenate([self.train_data, self.train_data_to_add], axis=1)

            test_responses = pickle.load(open(og_path + "/test_responses.pkl", "rb"))
            test_responses_to_add = pickle.load(open(added_path + "/test_responses.pkl", "rb"))

            self.test_data, self.test_labels, self.test_log_probs, \
                self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)
            
            self.test_data_to_add,_,_,_,_,_,_ = self.process_responses(test_responses_to_add)
            self.test_data = np.concatenate([self.test_data, self.test_data_to_add], axis=1)


    def process_data(self, split):

        all_responses = []

        if split == "train":
            questions = self.train_questions
            answers = self.train_answers

        else:
            questions = self.test_questions
            answers = self.test_answers

        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY,
        )

        # loop through questions 
        for q_ind, q in tqdm(enumerate(questions), total=len(questions)):
            
            to_add = {}
            response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": "Please answer the following question. " + "\ Q: \"{" + q + "}\"?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            generation, log_prob_dict = parse_response(response)
            to_add["generation"] = generation
            to_add["log_prob_dict"] = log_prob_dict
            to_add["answer"] = answers[q_ind]

            # get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": "Please answer the following question. " + "\ Q: \"{" + q + "}\"?"},
                    {"role": "user", "content": "Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": "Please answer the following question. " + "\ Q: \"{" + q + "}\"?"},
                    {"role": "assistant", "content": "Answer: " + generation},
                    {"role": "user", "content": "Did you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )

            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict
            
            # get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "user", "content": "Please answer the following question. " + "\ Q: \"{" + q + "}\"?"},
                        {"role": "assistant", "content": "Answer: " + generation},
                        {"role": "user", "content": "Now answer the following question to explain your decision (answer with yes or no). Question: " + exp },
                    ],
                    logprobs=True,
                    top_logprobs=5,
                )

                exp_generation, exp_log_prob_dict = parse_response(exp_response)
                to_add["exp_generation_" + str(ind)] = exp_generation
                to_add["exp_log_prob_dict_" + str(ind)] = exp_log_prob_dict

            all_responses.append(to_add)

        return all_responses
    
    def process_responses(self, responses):

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_labels = []
        sorted_logits = []
        all_logits = []

        for response in responses:
            generation = response["generation"]
            log_prob_dict = response["log_prob_dict"]
            answer = response["answer"]
            
            probs = get_yes_no_dist(log_prob_dict)
            all_log_probs.append(probs[1])
            
            # remove punctuation in generation
            generation = generation.replace("\"", "").replace("\'", "")

            # check if answer is in generation - answer is a list of multiple potential options
            if any([ans.lower() in generation.lower() for ans in answer]):
                all_labels.append(1)
            else:
                all_labels.append(0)

            # get sorted log prob values
            log_probs = np.array(list(log_prob_dict.values()))
            log_probs = np.sort(log_probs)[::-1]
            sorted_logits.append(log_probs)

            # map word in log_prob_dict to index in tokenizer vocab
            logits = np.zeros(len(tokenizer.get_vocab()))
            for token, log_prob in log_prob_dict.items():
                token_id = tokenizer.encode(token)[0]
                logits[token_id] = log_prob
            all_logits.append(logits)
            
            # get pre confidence
            pre_log_prob_dict = response["pre_log_prob_dict"]


            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(pre_log_prob_dict)
            all_pre_confs.append(probs[1])

            # get post confidence
            post_log_prob_dict = response["post_log_prob_dict"]

            # get probability of "yes" vs "no"
            probs = get_yes_no_dist(post_log_prob_dict)
            all_post_confs.append(probs[1])
            
            # get explanation responses
            exp_responses = np.zeros(len(self.explanation_prompts))
            for ind, exp in enumerate(self.explanation_prompts):
                exp_log_prob_dict = response["exp_log_prob_dict_" + str(ind)]
                exp_probs = get_yes_no_dist(exp_log_prob_dict)
                exp_responses[ind] = exp_probs[1]
            all_data.append(exp_responses)

        all_data = np.array(all_data)
        all_log_probs = np.array(all_log_probs)
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        sorted_logits = np.array(sorted_logits)
        return all_data, all_labels, all_log_probs, all_pre_confs, all_post_confs, all_logits, sorted_logits