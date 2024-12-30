import torch
import numpy as np
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import os

class BooIQDataset(torch.utils.data.Dataset):

    def __init__(self, path="/home/dylansam/research/llm_explanations/data/datasets/BooIQ/", split="train", \
                 tokenizer=None, tokenize=False, template=True):
        '''
        Dataset for BooIQ
        path - path to data directory
        split - train or val
        tokenizer - tokenizer to use for template
        tokenize - whether to tokenize the data
        '''
        
        self.tokenizer = tokenizer
        self.template = template
        
        if split == "train":
            questions, answers = self.process_data(path + "train.jsonl")
        elif split == "val" or split == "test":
            questions, answers = self.process_data(path + "dev.jsonl")

        if not tokenize:
            self.questions = questions
        else:
            tokenized_qs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids
            self.questions = tokenized_qs
        
        self.answers = answers
        self.options = ["False", "True"]

    def process_data(self, path):
        questions = []
        answers = []
        
        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)

                question = data["question"]
                passage = data["passage"]
                answer = data["answer"]

                # format passage and question
                prompt = "Answer the following question, with the following information. Information: {} \n \
                            Question: {}".format(passage, question)
                if self.template:
                    prompt_messages = [
                        {"role": "user", "content": prompt},
                    ]
                    prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                
                questions.append(prompt)

                if answer:
                    answers.append(1)
                else:
                    answers.append(0)

        return questions, answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

class CommonsenseQADataset(torch.utils.data.Dataset):

    def __init__(self, path="/home/dylansam/research/llm_explanations/data/datasets/CommonsenseQA/", split="train", \
                 tokenizer=None, tokenize=False, template=True):
        '''
        Dataset for CommonsenseQA
        path - path to data directory
        split - train or val
        tokenizer - tokenizer to use for template
        tokenize - whether to tokenize the data
        '''
        
        self.tokenizer = tokenizer
        self.template = template
        
        if split == "train":
            questions, answers = self.process_data(path + "train_rand_split.jsonl")
        elif split == "val" or split == "test":
            questions, answers = self.process_data(path + "dev_rand_split.jsonl")

        if not tokenize:
            self.questions = questions
        else:
            tokenized_qs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids
            self.questions = tokenized_qs

        self.answers = answers

    def process_data(self, path, template=True):
        questions = []
        answers = []
        self.options = ["A", "B", "C", "D", "E"]

        with open(path, "r") as f:
            for line in f:
                data = json.loads(line)

                question = data["question"]["stem"]
                choices = data["question"]["choices"]
                choices = "A: {}, B: {}, C: {}, D: {}, E: {}. ".format(choices[0]["text"], choices[1]["text"], choices[2]["text"], choices[3]["text"], choices[4]["text"])
                answer = data["answerKey"]

                # format question
                prompt = "Answer the following multiple choice question, and provide the correct answer, denoted by a letter in (A, B, C, D, E). \n \
                            Q: {} Potential Answers: {}".format(question, choices)
                
                if self.template:
                    prompt_messages = [
                        {"role": "user", "content": prompt},
                    ]
                    prompt = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                
                questions.append(prompt)

                if answer == "A":
                    answers.append(0)
                elif answer == "B":
                    answers.append(1)
                elif answer == "C":
                    answers.append(2)
                elif answer == "D":
                    answers.append(3)
                elif answer == "E":
                    answers.append(4)

        return questions, answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

class WinoGrandeDataset(torch.utils.data.Dataset):

    def __init__(self, split="train", tokenizer=None, tokenize=False, template=False):
        
        self.dataset = load_dataset("winogrande", "winogrande_debiased")[split]
        self.tokenizer = tokenizer
        self.template = template

        # check if path exists        
        questions, answers, options1, options2 = self.process_data()

        if not tokenize:
            self.questions = questions
            self.options1 = options1
            self.options2 = options2

        else:
            tokenized_qs = self.tokenizer(questions, return_tensors="pt", padding=True).input_ids
            self.questions = tokenized_qs
            tokenized_ops1 = self.tokenizer(options1, return_tensors="pt", padding=True).input_ids
            self.options1_tokens = tokenized_ops1
            tokenized_ops2 = self.tokenizer(options2, return_tensors="pt", padding=True).input_ids
            self.options2_tokens = tokenized_ops2

        self.answers = [x - 1 for x in answers]
        self.options = ["A", "B"]

    def process_data(self):

        questions = []
        answers = []
        options1 = []
        options2 = []

        for data in self.dataset:
            question = data["sentence"]
            answer = data["answer"]

            question = f"Fill in the blank (_) with the correct answer between the following options (A: {data['option1']}, B: {data['option2']}). Please answer with A or B. \n \
                        Q: {question}"
            if self.template:
                prompt_messages = [
                    {"role": "user", "content": question},
                ]
                question = self.tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
            
            questions.append(question)
            answers.append(int(answer))
            options1.append(data["option1"])
            options2.append(data["option2"])

        return questions, answers, options1, options2
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx], self.options1[idx], self.options2[idx]

class HateSpeechDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer=None, split="train", template=True):
        
        with open("/home/dylansam/research/llm_explanations/data/datasets/DHate/dhate.csv", "r") as f:
            data = f.readlines()
        
        train_strings, test_strings = [], []
        train_labels, test_labels = [], []

        for line in data[1:]:

            line = line.strip().split(",")

            if len(line) < 5: # skip last line or other errored lines
                continue

            split = line[-5].strip().replace("\"", "")
            
            if len(line) == 13: # no commas in text
                string = line[3].replace("\"", "")
                label = line[4].strip().replace("\"", "")
                label = 1 if label == "hate" else 0

            elif len(line) > 14:
                num_commas = len(line) - 13
                text = ",".join(line[3:3+num_commas+1]).replace("\"", "")
                string = text + line[3+num_commas].replace("\"", "")
                label = line[3+num_commas+1].strip().replace("\"", "")
                label = 1 if label == "hate" else 0

            else: # other weird formatting issue
                continue

            if split == "train":
                train_strings.append(string)
                train_labels.append(label)

            elif split == "test":
                test_strings.append(string)
                test_labels.append(label)

        # applying tokenizer template
        self.train_questions = []
        for question in train_strings:
            if template:
                prompt_messages = [
                    {"role": "user", "content": "Does the following text contain toxic content: \"" + question + "\"?"},
                ]
                self.train_questions.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
            else:
                prompt = "Does the following text contain toxic content: \"" + question + "\"?"
                self.train_questions.append(prompt)
        
        self.test_questions = []
        for question in test_strings:
            if template:
                prompt_messages = [
                    {"role": "user", "content": "Does the following text contain toxic content: \"" + question + "\"?"},
                ]
                self.test_questions.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
            else:
                prompt = "Does the following text contain toxic content: \"" + question + "\"?"
                self.test_questions.append(prompt)
        
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.options = ["no", "yes"]

        if split == "train":
            self.questions = self.train_questions
            self.answers = self.train_labels
        else:
            self.questions = self.test_questions
            self.answers = self.test_labels

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

class NQOpenDataset(torch.utils.data.Dataset):
    '''
    Open-ended questions from the Natural Questions dataset
    '''

    def __init__(self, split="train", tokenizer=None, tokenize=False, template=False):
        '''
        Data for Natural Questions - open ended subset with short answer format
        split - train or val
        tokenizer - tokenizer to use
        tokenize - whether to tokenize the data
        '''
        
        self.tokenizer = tokenizer
        self.dataset = load_dataset("nq_open")[split]
        questions = [ f"Answer the following question in a few words: \n \
                            Q: {q}?" for q in self.dataset["question"]]
        self.questions = []
        self.template = template
        for q in questions:
            if self.template:
                prompt_messages = [
                    {"role": "user", "content": q}
                ]
                self.questions.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
            else:
                self.questions.append(q)
        self.answers = self.dataset["answer"]

        if tokenize:
            tokenized_qs = self.tokenizer(self.questions, return_tensors="pt", padding=True).input_ids
            self.questions = tokenized_qs
            tokenized_ans = []
            for ans in self.answers:
                to_ret = []
                for a in ans:
                    to_ret.append(self.tokenizer(a, return_tensors="pt").input_ids)
                tokenized_ans.append(to_ret)
            self.answers = tokenized_ans

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

class HaluEvalDataset(torch.utils.data.Dataset):

    def __init__(self, 
            root="/home/dylansam/research/llm_explanations/data/datasets/halu_eval_data", tokenizer=None, 
            split="train", tokenize=False, template=False):
        '''
        Root - path to data directory
        tokenizer - what tokenizer to use
        tokenize - whether to tokenize the data
        '''

        self.root = root
        self.tokenizer = tokenizer

        # load data
        data = []
        with open(os.path.join(root, f"general_data.json"), "r") as f:
            for line in f:
                data.append(json.loads(line))
        
        self.questions = []
        for d in data:
            query = d["user_query"]
            answer = d["chatgpt_response"]
            if template:
                prompt_messages = [
                    {"role": "user", "content": "Does the following query and response contain a hallucination?: \" Query: " + query + " Response: " + answer + "\"?"},
                ]
                self.questions.append(
                    tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
                )
            else:
                self.questions.append(
                    "Does the following query and response contain a hallucination?: \" Query: " + query + " Response: " + answer + "\"?"
                )

        self.answers = [1 if d["hallucination"] == "yes" else 0 for d in data]
        self.options = ["no", "yes"]

        if split == "train":
            self.questions = self.questions[:3500]
            self.answers = self.answers[:3500]
        
        else:
            self.questions = self.questions[3500:]
            self.answers = self.answers[3500:]
        

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx], self.labels[idx]

class SquadDataset(torch.utils.data.Dataset):

    def __init__(self, split="train", tokenizer=None, tokenize=False):
        
        '''
        Squad v1.1 dataset
        split - train or val
        tokenizer - tokenizer to use
        tokenize - whether to tokenize the data
        '''

        self.tokenizer = tokenizer
        self.dataset = load_dataset("squad")[split]
        questions = ["[INST] Please answer the following question using only a sequence of text from the following context. Context: \"{}\" \n \
                                Q: \"{}\"? [\INST] Answer:".format(q["context"], q["question"]) for q in self.dataset]
        self.questions = []
        # for q in questions:
            # prompt_messages = [
                # {"role": "user", "content": q}
            # ]
            # self.questions.append(tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True))
        self.questions = questions
        self.answers = [q["answers"]["text"][0] for q in self.dataset]

        if tokenize:
            self.tokenizer = tokenizer
            tokenized_qs = self.tokenizer(self.questions, return_tensors="pt", padding=True).input_ids
            self.questions = tokenized_qs

            tokenized_ans = []
            for ans in self.answers:
                to_ret = []
                for a in ans:
                    to_ret.append(self.tokenizer(a, return_tensors="pt").input_ids)
                tokenized_ans.append(to_ret)
            self.answers = tokenized_ans

    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("/data/locus/project_data/project_data2/dylansam/Llama-2-7b-hf/")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = BooIQDataset(split="train", tokenizer=tokenizer)
    print(dataset[0])
    print(len(dataset))