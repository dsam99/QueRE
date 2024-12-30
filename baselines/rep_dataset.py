import torch
import numpy as np
from tqdm import tqdm
import os
from src.llm import load_llm, get_left_pad, get_add_token
from data.dataset import BooIQDataset, CommonsenseQADataset, WinoGrandeDataset, NQOpenDataset, HaluEvalDataset, HateSpeechDataset, SquadDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set seeds
np.random.seed(0)
torch.manual_seed(0)

class RepDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, model_type):

        self.dataset = dataset
        self.model_type = model_type
        self.left_pad = get_left_pad(model_type)
        self.add_token = get_add_token(model_type)
        
        rep_path = "./data/rep_dataset/" + dataset + "/" + model_type + "/"
        if not os.path.exists(rep_path):
            os.makedirs(rep_path)

        self.context_examples = None
        if not os.path.exists(rep_path + "train_rep.npy") or not os.path.exists(rep_path + "test_rep.npy"):
            
            self.model, self.tokenizer = load_llm(model_type)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if os.path.exists(rep_path + "train_rep.npy"):
                self.train_rep = np.load(rep_path + "train_rep.npy")
            else:
                print(f"Creating rep_dataset at {rep_path}")
                self.train_rep = self.create_rep_dataset(split="train")
                np.save(rep_path + "train_rep.npy", self.train_rep)
            
            if os.path.exists(rep_path + "test_rep.npy"):
                self.test_rep = np.load(rep_path + "test_rep.npy")
            else:
                self.test_rep = self.create_rep_dataset(split="test")
                np.save(rep_path + "test_rep.npy", self.test_rep)
        
        else:
            self.train_rep = np.load(rep_path + "train_rep.npy")
            self.test_rep = np.load(rep_path + "test_rep.npy")

        if "70b" in model_type:
            self.train_rep = self.train_rep[:1000]

    def create_rep_dataset(self, split="train"):
        
        reps = [] 

        test_map = {
            "squad": "validation",
            "WinoGrande": "validation",
            "nq": "validation",
        }

        # remap split if necessary for valid
        if split == "test" and self.dataset in test_map:
            split = test_map[self.dataset]

        if self.dataset == "CommonsenseQA":
            dataset = CommonsenseQADataset(split=split, tokenizer=self.tokenizer)
        elif self.dataset == "WinoGrande":
            dataset = WinoGrandeDataset(split=split, tokenizer=self.tokenizer)
        elif self.dataset == "BooIQ":
            dataset = BooIQDataset(split=split, tokenizer=self.tokenizer)
        elif self.dataset == "nq":

            train_dataset = NQOpenDataset(split="train", tokenizer=self.tokenizer)
            dataset = NQOpenDataset(split=split, tokenizer=self.tokenizer)

            # take first 2 examples as context
            num_context = 2
            self.context_examples = ""

            for i in range(num_context):
                self.context_examples += train_dataset.questions[i] + " " + train_dataset.answers[i][0] + "\n"

            if split == "train":
                dataset.questions = dataset.questions[2:]

        elif self.dataset == "ToxicEval":
            dataset = HateSpeechDataset(split=split, tokenizer=self.tokenizer)
        elif self.dataset == "squad":
            dataset = SquadDataset(split=split, tokenizer=self.tokenizer)
        elif self.dataset == "HaluEval":
            dataset = HaluEvalDataset(split=split, tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Dataset {self.dataset} not recognized")

        # evaluate on a subset
        if split == "train":

            if "70b" in self.model_type:
                subset = min(1000, len(dataset.questions))
            else:
                subset = min(5000, len(dataset.questions))
        else:
            
            subset = min(1000, len(dataset.questions))

        for i in tqdm(range(subset), total=subset):

            if self.context_examples is not None:
                input_string = self.context_examples + dataset.questions[i]
            else:
                input_string = dataset.questions[i]

            input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(device)
            with torch.no_grad():
                output_dict = self.model(input_ids, return_dict=True, output_hidden_states=True)
            hidden_states = output_dict.hidden_states
            rep = hidden_states[-1][0,-1].detach().cpu().numpy()
            reps.append(rep)

        reps = np.array(reps)
        return reps

if __name__ == "__main__":

    dataset = "nq-open"
    rep_dataset = RepDataset(dataset, "llama-7b")
    print(rep_dataset.train_rep.shape)
    print(rep_dataset.test_rep.shape)