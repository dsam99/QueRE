from src.quere import ClosedEndedExplanationDataset, OpenEndedExplanationDataset, SquadExplanationDataset
import argparse
import numpy as np
import torch

if __name__ == "__main__":

    # set random seed
    np.random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama-7b")
    parser.add_argument("--dataset", type=str, default="WinoGrande")
    parser.add_argument("--random", action="store_true", default=False, help="Use random prompts")
    parser.add_argument("--gpt_exp", action="store_true", default=False, help="Use GPT explanations")
    parser.add_argument("--gpt_diverse", action="store_true", default=False, help="Use diverse GPT explanations")
    parser.add_argument("--gpt_sim", action="store_true", default=False, help="Use GPT explanations with similar prompts")
    parser.add_argument("--random_tokens", action="store_true", default=False, help="Use random tokens")

    args = parser.parse_args()

    if args.dataset == "BooIQ":
        dataset = ClosedEndedExplanationDataset("BooIQ", args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
    elif args.dataset == "HaluEval":
        dataset = ClosedEndedExplanationDataset("HaluEval", args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
    elif args.dataset == "ToxicEval":
        dataset = ClosedEndedExplanationDataset("ToxicEval", args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
    elif args.dataset == "CommonsenseQA":
        dataset = ClosedEndedExplanationDataset("CommonsenseQA", args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
    elif args.dataset == "WinoGrande":
        dataset = ClosedEndedExplanationDataset("WinoGrande", args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
    elif args.dataset == "squad":
        dataset = SquadExplanationDataset(args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
    elif args.dataset == "nq":
        dataset = OpenEndedExplanationDataset(args.llm, gpt_exp=args.gpt_exp, gpt_diverse=args.gpt_diverse, random=args.random, gpt_sim=args.gpt_sim, random_tokens=args.random_tokens)
