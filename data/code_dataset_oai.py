import jsonlines
import os
from datasets import load_dataset
from src.utils import (
    get_preconf_prompt, get_postconf_prompt, explanation_prompts, gpt_explanation_prompts, 
    get_adv_code_sys_prompt, get_adv_code_sys_prompt2, get_adv_code_sys_prompt3,
    parse_response, get_yes_no_dist, get_answer_dist, get_syc_code_sys_prompt
)
import numpy as np
import torch
from tqdm import tqdm
from openai import OpenAI
import pickle
from transformers import GPT2Tokenizer

valid_models = [
    "gpt-3.5-turbo-0125",
    "gpt-4o-mini",
]

# Change to your own organization and project
ORG="<enter_org_here>"
PROJECT="<enter_project_here>"
KEY=os.environ.get("OPENAI_API_KEY_QUERE")

# set seeds
np.random.seed(0)
torch.manual_seed(0)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def generate_code_completions(model, system_prompt="adv"):
    """
    Generate code completions for the deepmind/code_contests dataset.
    """

    dataset = load_dataset("deepmind/code_contests")

    client = OpenAI(
        organization=ORG,
        project=PROJECT,
        api_key=KEY,
    )

    train_code_prompts = dataset["train"]["description"][:500]
    test_code_prompts = dataset["test"]["description"]

    # adversarial system prompt
    if system_prompt == "adv":
        adv_code_prompt = get_adv_code_sys_prompt()
        out_path = "./data/quere_datasets/adversarial_code/" + model + "/"
    elif system_prompt == "adv2":
        adv_code_prompt = get_adv_code_sys_prompt2()
        out_path = "./data/quere_datasets/adversarial_code2/" + model + "/"
    elif system_prompt == "adv3":
        adv_code_prompt = get_adv_code_sys_prompt3()
        out_path = "./data/quere_datasets/adversarial_code3/" + model + "/"
    elif system_prompt == "syc":
        adv_code_prompt = get_syc_code_sys_prompt()
        out_path = "./data/quere_datasets/adversarial_code_syc/" + model + "/"

    # get code generations
    train_generations = []
    test_generations = []

    train_adv_generations = []
    test_adv_generations = []
    
    for ind, prompt in tqdm(enumerate(train_code_prompts), total = len(train_code_prompts)):
        
        exp_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Please generate code for the following prompt. Please only respond with code and no other explanations. {}".format(prompt)},
            ],
            max_tokens=200,
        )

        adv_exp_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": adv_code_prompt},
                {"role": "user", "content": "Please generate code for the following prompt. Please only respond with code and no other explanations. {}".format(prompt)},
            ],
            max_tokens=200,
        )
        train_generations.append(exp_response.choices[0].message.content)
        train_adv_generations.append(adv_exp_response.choices[0].message.content)
    
    for ind, prompt in tqdm(enumerate(test_code_prompts), total = len(test_code_prompts)):
            
            exp_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Please generate code for the following prompt. Please only respond with code and no other explanations. {}".format(prompt)},
                ],
                max_tokens=200,
            )
    
            adv_exp_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": adv_code_prompt},
                    {"role": "user", "content": "Please generate code for the following prompt. Please only respond with code and no other explanations. {}".format(prompt)},
                ],
                max_tokens=200,
            )
    
            test_generations.append(exp_response.choices[0].message.content)
            test_adv_generations.append(adv_exp_response.choices[0].message.content)
    
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

def parse_code_completion_data(data_path, adv=False):

    train_data = []
    test_data = []
        
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

    def __init__(self, model_type, adv=False, adv2=False, adv3=False, syc=False):
        self.model_type = model_type
        self.adv = adv
        self.adv2 = adv2
        self.adv3 = adv3
        self.syc = syc

        if adv:
            folder_path = "./data/quere_datasets/adv_code/" + model_type
        elif adv2:
            folder_path = "./data/quere_datasets/adv_code2/" + model_type
        elif adv3:
            folder_path = "./data/quere_datasets/adv_code3/" + model_type
        elif syc: # sycophantic system prompt - i think the answer includes dynamic programming
            folder_path = "./data/quere_datasets/adv_code_syc/" + model_type
        else:
            folder_path = "./data/quere_datasets/adv_code_clean/" + model_type

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        path = os.path.join(folder_path, "train_responses.pkl")

        # Load code completion data
        if adv:
            train_data, test_data = parse_code_completion_data("./data/quere_datasets/adversarial_code/" + model_type + "/", adv=True)
        elif adv2:
            train_data, test_data = parse_code_completion_data("./data/quere_datasets/adversarial_code2/" + model_type + "/", adv=True)
        elif adv3:
            train_data, test_data = parse_code_completion_data("./data/quere_datasets/adversarial_code3/" + model_type + "/", adv=True)
        elif syc:
            train_data, test_data = parse_code_completion_data("./data/quere_datasets/adversarial_code_syc/" + model_type + "/", adv=True)
        else:
            train_data, test_data = parse_code_completion_data("./data/quere_datasets/adversarial_code/" + model_type + "/", adv=False)

        self.train_prompts = [d[0] for d in train_data]
        self.train_generations = [d[1] for d in train_data]

        self.test_prompts = [d[0] for d in test_data]
        self.test_generations = [d[1] for d in test_data]

        # Get prompts
        self.preconf_prompt = get_preconf_prompt()
        self.postconf_prompt = get_postconf_prompt()
        self.explanation_prompts = explanation_prompts() + gpt_explanation_prompts()

        if not os.path.exists(path) or not os.path.exists(os.path.join(folder_path, "test_responses.pkl")):
            print("No data found. Generating new data...")

            if os.path.exists(path):
                print("Loading existing train data...")
                train_responses = pickle.load(open(path, "rb"))
            
            else:
                if self.adv:
                    train_responses = self.process_data_sys_prompt('train', get_adv_code_sys_prompt())
                elif self.adv2:
                    train_responses = self.process_data_sys_prompt('train', get_adv_code_sys_prompt2())
                elif self.adv3:
                    train_responses = self.process_data_sys_prompt('train', get_adv_code_sys_prompt3())
                elif self.syc:
                    train_responses = self.process_data_sys_prompt('train', get_syc_code_sys_prompt())
                else:
                    train_responses = self.process_data('train')
                
                pickle.dump(train_responses, open(path, "wb"))

            self.train_data, self.train_labels, self.train_log_probs, self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            test_responses_path = os.path.join(folder_path, "test_responses.pkl")
            if os.path.exists(test_responses_path):
                print("Loading existing test data...")
                test_responses = pickle.load(open(test_responses_path, "rb"))
            else:
                if self.adv:
                    test_responses = self.process_data_sys_prompt('test', get_adv_code_sys_prompt())
                elif self.adv2:
                    test_responses = self.process_data_sys_prompt('test', get_adv_code_sys_prompt2())
                elif self.adv3:
                    test_responses = self.process_data_sys_prompt('test', get_adv_code_sys_prompt3())
                elif self.syc:
                    test_responses = self.process_data_sys_prompt('train', get_syc_code_sys_prompt())
                else:
                    test_responses = self.process_data('test')
                pickle.dump(test_responses, open(test_responses_path, "wb"))

            self.test_data, self.test_labels, self.test_log_probs, self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

        else:
            # Load existing data
            train_responses = pickle.load(open(path, "rb"))
            self.train_data, self.train_labels, self.train_log_probs, self.train_pre_confs, self.train_post_confs, self.train_logits, self.train_sorted_logits = self.process_responses(train_responses)

            test_responses = pickle.load(open(os.path.join(folder_path, "test_responses.pkl"), "rb"))
            self.test_data, self.test_labels, self.test_log_probs, self.test_pre_confs, self.test_post_confs, self.test_logits, self.test_sorted_logits = self.process_responses(test_responses)

    def process_data(self, split):
        all_responses = []

        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY
        )

        if split == 'train':
            prompts = self.train_prompts
            generations = self.train_generations
        else:
            prompts = self.test_prompts
            generations = self.test_generations

        for idx in tqdm(range(len(prompts)), total=len(prompts)):
            prompt = prompts[idx]
            generation = generations[idx]

            to_add = {}
            to_add['prompt'] = prompt
            to_add['generation'] = generation

            # Get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": prompt + " Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )
            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # Get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": generation},
                    {"role": "user", "content": self.postconf_prompt},
                ],
                logprobs=True,
                top_logprobs=5,
            )
            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict

            # Get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": generation},
                        {"role": "user", "content": f"Now answer the following question to explain your code (answer with yes or no). Question: {exp}"},
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

        client = OpenAI(
            organization=ORG,
            project=PROJECT,
            api_key=KEY
        )

        if split == 'train':
            prompts = self.train_prompts
            generations = self.train_generations
        else:
            prompts = self.test_prompts
            generations = self.test_generations

        for idx in tqdm(range(len(prompts)), total=len(prompts)):
            prompt = prompts[idx]
            generation = generations[idx]

            to_add = {}
            to_add['prompt'] = prompt
            to_add['generation'] = generation

            # Get pre confidence
            pre_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt + " Will you answer this question correctly (answer with yes or no)?"},
                ],
                logprobs=True,
                top_logprobs=5,
            )
            pre_generation, pre_log_prob_dict = parse_response(pre_conf_response)
            to_add["pre_generation"] = pre_generation
            to_add["pre_log_prob_dict"] = pre_log_prob_dict

            # Get post confidence
            post_conf_response = client.chat.completions.create(
                model=self.model_type,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": generation},
                    {"role": "user", "content": self.postconf_prompt},
                ],
                logprobs=True,
                top_logprobs=5,
            )
            post_generation, post_log_prob_dict = parse_response(post_conf_response)
            to_add["post_generation"] = post_generation
            to_add["post_log_prob_dict"] = post_log_prob_dict

            # Get explanation responses
            for ind, exp in enumerate(self.explanation_prompts):
                exp_response = client.chat.completions.create(
                    model=self.model_type,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": generation},
                        {"role": "user", "content": f"Now answer the following question to explain your code (answer with yes or no). Question: {exp}"},
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
        all_pre_confs = []
        all_post_confs = []
        sorted_logits = []
        all_log_probs = []
        all_logits = []

        for response in responses:
            generation = response["generation"]
            log_prob_dict = response["pre_log_prob_dict"] # using pre for overall since no answers
            log_probs = np.array(list(log_prob_dict.values()))
            log_probs = np.sort(log_probs)[::-1]
            all_log_probs.append(log_probs)
            
            # remove punctuation in generation
            generation = generation.replace("\"", "").replace("\'", "")

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
        all_logits = np.array(all_logits)
        sorted_logits = np.array(sorted_logits)
        all_labels = [] # no labels for this task

        return all_data, all_labels, all_log_probs, all_pre_confs, all_post_confs, all_logits, sorted_logits


if __name__ == "__main__":
    generate_code_completions("gpt-4o-mini", system_prompt="adv2")
    generate_code_completions("gpt-4o-mini", system_prompt="adv3")