import jsonlines
import os
from datasets import load_dataset
from src.utils import (
    get_preconf_prompt, get_postconf_prompt, explanation_prompts, gpt_explanation_prompts, 
    get_adv_code_sys_prompt, get_adv_code_sys_prompt2, get_adv_code_sys_prompt3,
    parse_response,
)
from src.llm import load_llm, get_left_pad, get_add_token
import numpy as np
import torch
from tqdm import tqdm
import pickle
import gc

# set seeds
np.random.seed(0)
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_code_completions(model):

    dataset = load_dataset("deepmind/code_contests")

    train_code_prompts = dataset["train"]["description"][:500]
    test_code_prompts = dataset["test"]["description"]

    # adversarial system prompt
    adv_code_prompt = get_adv_code_sys_prompt()
    out_path = "./data/adversarial_code/" + model + "/"

    # get code generations
    train_generations = []
    test_generations = []

    train_adv_generations = []
    test_adv_generations = []
    llm, tokenizer = load_llm(model)

    def generate_response(prompt, adv=False):
        if adv:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "system", "content": adv_code_prompt}, {"role": "user", "content": prompt}],
                tokenize=False
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False
            )
        inputs = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        inputs_length = inputs.shape[1]
        outputs = llm.generate(
            inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id
        )
        generated_tokens = outputs[0][inputs_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return generated_text

    for ind, prompt in tqdm(enumerate(train_code_prompts), total = len(train_code_prompts)):

        # normal generation
        generation = generate_response(prompt, adv=False)
        train_generations.append(generation)

        # adversarial generation
        adv_generation = generate_response(prompt, adv=True)
        train_adv_generations.append(adv_generation)

    
    for ind, prompt in tqdm(enumerate(test_code_prompts), total = len(test_code_prompts)):
            
        # normal generation
        generation = generate_response(prompt, adv=False)
        test_generations.append(generation)

        # adversarial generation
        adv_generation = generate_response(prompt, adv=True)
        test_adv_generations.append(adv_generation)
    
    print("Train generations: ", train_generations)

    # write to file
    with jsonlines.open(out_path + "train_generations.jsonl", "w") as writer:
        for i in range(len(train_generations)):
            writer.write(
                {
                    "prompt": train_code_prompts[i],
                    "generation": train_generations[i],
                    "adv_generation": train_adv_generations[i],
                })
    
    with jsonlines.open(out_path + "test_generations.jsonl", "w") as writer:
        for i in range(len(test_generations)):
            writer.write(
                {
                    "prompt": test_code_prompts[i],
                    "generation": test_generations[i],
                    "adv_generation": test_adv_generations[i],
                })

def parse_code_completion_data(model_type, adv=False):

    train_data = []
    test_data = []

    data_path = "./data/adversarial_code/" + model_type + "/"
        
    with jsonlines.open(data_path + "train_generations.jsonl") as reader:
        for obj in reader:
            prompt = obj["prompt"]
            if adv:
                generation = obj["adv_generation"]
            else:
                generation = obj["generation"]
            train_data.append((prompt, generation))
    
    with jsonlines.open(data_path + "test_generations.jsonl") as reader:
        for obj in reader:
            prompt = obj["prompt"]
            if adv:
                generation = obj["adv_generation"]
            else:
                generation = obj["generation"]
            test_data.append((prompt, generation))
    
    return train_data, test_data


class AdversarialCodeDataset(torch.utils.data.Dataset):
    """
    Dataset to distinguish between normal and adversarial code generation.
    Uses deepmind/code_contests dataset.
    """

    def __init__(self, model_type, adv=False, adv2=False, adv3=False):
        self.model_type = model_type
        self.adv = adv
        self.adv2 = adv2
        self.adv3 = adv3

        if adv:
            folder_path = "./data/adv_code/" + model_type
        elif adv2:
            folder_path = "./data/adv_code2/" + model_type
        elif adv3:
            folder_path = "./data/adv_code3/" + model_type
        else:
            folder_path = "./data/adv_code_clean/" + model_type

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, "train_responses.pkl")

        # Load code completion data
        train_data, test_data = parse_code_completion_data(model_type, adv=adv)
        self.train_prompts = [d[0] for d in train_data]
        self.train_generations = [d[1] for d in train_data]

        self.test_prompts = [d[0] for d in test_data]
        self.test_generations = [d[1] for d in test_data]

        # Get prompts
        self.preconf_prompt = get_preconf_prompt()
        self.postconf_prompt = get_postconf_prompt()
        self.explanation_prompts = explanation_prompts() + gpt_explanation_prompts()

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = folder_path + "/train_explanations.npy"
        if not os.path.exists(path) or not os.path.exists(folder_path + "/test_explanations.npy"):
            print("No data found. Generating new data...")

            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            self.model_type = model_type
            self.left_pad = get_left_pad(model_type)
            self.add_token = get_add_token(model_type)

            # resulting arrays
            self.train_data, self.test_data = [], []
            self.train_log_probs, self.test_log_probs = [], []
            self.train_pre_confs, self.test_pre_confs = [], []
            self.train_post_confs, self.test_post_confs = [], []
            self.train_logits, self.test_logits = [], []

        if os.path.exists(path):
            self.train_data = np.load(folder_path + "/train_explanations.npy")
            self.train_labels = np.load(folder_path + "/train_labels.npy")
            self.train_log_probs = np.load(folder_path + "/train_log_probs.npy")
            self.train_pre_confs = np.load(folder_path + "/train_pre_confs.npy")
            self.train_post_confs = np.load(folder_path + "/train_post_confs.npy")
            self.train_logits = np.load(folder_path + "/train_logits.npy")

        else:
            if self.adv:
                self.train_data, self.train_log_probs, self.train_labels, self.train_pre_confs, \
                    self.train_post_confs, self.train_logits = self.process_data_sys_prompt('train', get_adv_code_sys_prompt())
            # elif self.adv2:
                # train_responses = self.process_data_sys_prompt('train', get_adv_code_sys_prompt2())
            # elif self.adv3:
                # train_responses = self.process_data_sys_prompt('train', get_adv_code_sys_prompt3())
            else:
                self.train_data, self.train_log_probs, self.train_labels, self.train_pre_confs, \
                    self.train_post_confs, self.train_logits  = self.process_data('train')            
            
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
            if self.adv:
                self.test_data, self.test_log_probs, self.test_labels, self.test_pre_confs, \
                    self.test_post_confs, self.test_logits = self.process_data_sys_prompt('test', get_adv_code_sys_prompt())
            # elif self.adv2:
                # test_responses = self.process_data_sys_prompt('test', get_adv_code_sys_prompt2())
            # elif self.adv3:
                # test_responses = self.process_data_sys_prompt('test', get_adv_code_sys_prompt3())
            else:
                self.test_data, self.test_log_probs, self.test_labels, self.test_pre_confs, \
                        self.test_post_confs, self.test_logits = self.process_data("test")
            
            np.save(folder_path + "/test_explanations.npy", self.test_data)
            np.save(folder_path + "/test_labels.npy", self.test_labels)
            np.save(folder_path + "/test_log_probs.npy", self.test_log_probs)
            np.save(folder_path + "/test_pre_confs.npy", self.test_pre_confs)
            np.save(folder_path + "/test_post_confs.npy", self.test_post_confs)
            np.save(folder_path + "/test_logits.npy", self.test_logits)
        
        # delete model
        # if self.llm is not None:
            # del self.model
            # gc.collect()

        # reshape log probs
        # self.train_log_probs = self.train_log_probs.reshape(-1, 2)
        # self.test_log_probs = self.test_log_probs.reshape(-1, 2)

        # convert labels from downstream task label to if model was correct
        # model_preds = np.argmax(self.train_log_probs, axis=1)
        # self.train_labels = (model_preds == self.train_labels).astype(int)

        # model_preds = np.argmax(self.test_log_probs, axis=1)
        # self.test_labels = (model_preds == self.test_labels).astype(int)


    def process_data(self, split):
        if split == "train":
            base_prompts = self.train_prompts
            base_generations = self.train_generations
        else:
            base_prompts = self.test_prompts
            base_generations = self.test_generations

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_logits = []
        all_labels = []            

        # get indices of yes and no tokens -> possible answers to question
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

        # loop through all prompts and generations
        for idx in tqdm(range(len(base_prompts)), total=len(base_prompts)):

            # get last token logits after question
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": base_prompts[idx]}, {"role": "assistant", "content": base_generations[idx]}],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()
            all_logits.append(logits)

            # Get pre confidence
            pre_conf_template = [
                {"role": "user", "content": base_prompts[idx] + self.preconf_prompt},
                # {"role": "assistant", "content": base_generations[idx]},
                # {"role": "user", "content": self.preconf_prompt},
            ]
            pre_conf_input = self.tokenizer.apply_chat_template(pre_conf_template, return_tensors="pt", add_generation_prompt=True).to(device)
            with torch.no_grad():
                logits = self.model(pre_conf_input, return_dict=True).logits[0, -1, :].cpu()
            pre_conf_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
            pre_conf_dist = torch.nn.functional.softmax(pre_conf_dist, dim=0)
            pre_conf = pre_conf_dist[0].item()
            all_pre_confs.append(pre_conf)

            # get post confidence
            post_conf_template = [
                {"role": "user", "content": base_prompts[idx]},
                {"role": "assistant", "content": base_generations[idx]},
                {"role": "user", "content": self.postconf_prompt},
            ]
            post_conf_input = self.tokenizer.apply_chat_template(post_conf_template, return_tensors="pt", add_generation_prompt=True).to(device)
            with torch.no_grad():
                logits = self.model(post_conf_input, return_dict=True).logits[0, -1, :].cpu()

            post_conf_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
            post_conf_dist = torch.nn.functional.softmax(post_conf_dist, dim=0)
            post_conf_n = post_conf_dist[0].item()
            all_post_confs.append(post_conf_n)

            # Get explanation responses
            # prob_dist = np.zeros((2*len(self.explanation_prompts),))
            prob_dist = np.zeros((len(self.explanation_prompts),))
            num_batch = 8

            for i in range(0, len(self.explanation_prompts)):
                # exp_inputs = [q + " " + ans + " " + exp for ans in ["yes", "no"] for exp in self.explanation_prompts[i:i+num_batch]]
                
                exp_input_template = [
                        {"role": "user", "content": base_prompts[idx]}, 
                        {"role": "assistant", "content": base_generations[idx]},
                        {"role": "user", "content": self.explanation_prompts[i]}
                ]
                input_ids = self.tokenizer.apply_chat_template(exp_input_template, return_tensors="pt", add_generation_prompt=True).to(device)

                with torch.no_grad():
                    logits = self.model(input_ids, return_dict=True).logits[0]
                    # last_token_id = token_dict.attention_mask.sum(1) - 1
                    last_token_id = input_ids.shape[0] - 1

                # get probability of yes (w.r.t. distribution [yes, no])
                if self.left_pad:
                    logits = logits[-1, :]
                else:
                    logits = logits[last_token_id, :]

                logits = torch.stack([logits[no_token_id], logits[yes_token_id]]).squeeze()
                # prob_dist[2 * i: 2 * (i + num_batch)] = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
                to_add_dist = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
                prob_dist[i] = to_add_dist[1]
                # del from memory
                del input_ids
                del logits

                gc.collect()

            all_data.append(prob_dist)  

        # reshaping
        all_data = np.array(all_data)
        # all_log_probs = np.array(all_log_probs) # no such thing
        all_log_probs = []
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_logits = np.array(all_logits)
        # all_labels = np.array(all_labels) # also no such thing
        all_labels = []

        return all_data, all_log_probs, all_labels, all_pre_confs, all_post_confs, all_logits
    
    def process_data_sys_prompt(self, split, system_prompt):
        if split == "train":
            base_prompts = self.train_prompts
            base_generations = self.train_generations
        else:
            base_prompts = self.test_prompts
            base_generations = self.test_generations

        all_data = []
        all_log_probs = []
        all_pre_confs = []
        all_post_confs = []
        all_logits = []
        all_labels = []            

        # get indices of yes and no tokens -> possible answers to question
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

        # loop through all prompts and generations
        for idx in tqdm(range(len(base_prompts)), total=len(base_prompts)):

            # get last token logits after question
            input_ids = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_prompts[idx]}, 
                    {"role": "assistant", "content": base_generations[idx]}
                ],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = self.model(input_ids, return_dict=True).logits[0, -1, :].cpu()
            all_logits.append(logits)

            # Get pre confidence
            pre_conf_template = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_prompts[idx] + self.preconf_prompt},
                # {"role": "assistant", "content": base_generations[idx]},
                # {"role": "user", "content": self.preconf_prompt},
            ]
            pre_conf_input = self.tokenizer.apply_chat_template(pre_conf_template, return_tensors="pt", add_generation_prompt=True).to(device)
            with torch.no_grad():
                logits = self.model(pre_conf_input, return_dict=True).logits[0, -1, :].cpu()
            pre_conf_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
            pre_conf_dist = torch.nn.functional.softmax(pre_conf_dist, dim=0)
            pre_conf = pre_conf_dist[0].item()
            all_pre_confs.append(pre_conf)

            # get post confidence
            post_conf_template = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": base_prompts[idx]},
                {"role": "assistant", "content": base_generations[idx]},
                {"role": "user", "content": self.postconf_prompt},
            ]
            post_conf_input = self.tokenizer.apply_chat_template(post_conf_template, return_tensors="pt", add_generation_prompt=True).to(device)
            with torch.no_grad():
                logits = self.model(post_conf_input, return_dict=True).logits[0, -1, :].cpu()

            post_conf_dist = torch.stack([logits[yes_token_id], logits[no_token_id]], dim=0).squeeze()
            post_conf_dist = torch.nn.functional.softmax(post_conf_dist, dim=0)
            post_conf_n = post_conf_dist[0].item()
            all_post_confs.append(post_conf_n)

            # Get explanation responses
            # prob_dist = np.zeros((2*len(self.explanation_prompts),))
            prob_dist = np.zeros((len(self.explanation_prompts),))
            num_batch = 8

            for i in range(0, len(self.explanation_prompts)):
                # exp_inputs = [q + " " + ans + " " + exp for ans in ["yes", "no"] for exp in self.explanation_prompts[i:i+num_batch]]
                
                exp_input_template = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": base_prompts[idx]}, 
                        {"role": "assistant", "content": base_generations[idx]},
                        {"role": "user", "content": self.explanation_prompts[i]}
                ]
                input_ids = self.tokenizer.apply_chat_template(exp_input_template, return_tensors="pt", add_generation_prompt=True).to(device)

                with torch.no_grad():
                    logits = self.model(input_ids, return_dict=True).logits[0]
                    # last_token_id = token_dict.attention_mask.sum(1) - 1
                    last_token_id = input_ids.shape[0] - 1

                # get probability of yes (w.r.t. distribution [yes, no])
                if self.left_pad:
                    logits = logits[-1, :]
                else:
                    logits = logits[last_token_id, :]

                logits = torch.stack([logits[no_token_id], logits[yes_token_id]]).squeeze()
                # prob_dist[2 * i: 2 * (i + num_batch)] = torch.nn.functional.softmax(logits, dim=1)[:, 1].cpu().numpy()
                to_add_dist = torch.nn.functional.softmax(logits, dim=0).cpu().numpy()
                prob_dist[i] = to_add_dist[1]
                # del from memory
                del input_ids
                del logits

                gc.collect()

            all_data.append(prob_dist)  

        # reshaping
        all_data = np.array(all_data)
        # all_log_probs = np.array(all_log_probs) # no such thing
        all_log_probs = []
        all_pre_confs = np.array(all_pre_confs)
        all_post_confs = np.array(all_post_confs)
        all_logits = np.array(all_logits)
        # all_labels = np.array(all_labels) # also no such thing
        all_labels = []

        return all_data, all_log_probs, all_labels, all_pre_confs, all_post_confs, all_logits
    

    

if __name__ == "__main__":
    AdversarialCodeDataset("llama-7b", adv=True)