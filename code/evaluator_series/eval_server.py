import os
import argparse
import pandas as pd
import torch
from evaluators.chatgpt import ChatGPT_Evaluator
from evaluators.moss import Moss_Evaluator
from evaluators.chatglm import ChatGLM_Evaluator
from evaluators.minimax import MiniMax_Evaluator
from evaluators.qwen_1_5 import Qwen_1_5_Evaluator
from evaluators.qwen_1_5_vllm import Vllm_Qwen_1_5_Evaluator
from evaluators.qwen_1_5_trtllm import Trtllm_Qwen_1_5_Evaluator

import json
import numpy as np

import time
choices = ["A", "B", "C", "D"]

def main(args):

    if "turbo" in args.model_name or "gpt-4" in args.model_name:
        evaluator=ChatGPT_Evaluator(
            choices=choices,
            k=args.ntrain,
            api_key=args.openai_key,
            model_name=args.model_name
        )
    elif "moss" in args.model_name:
        evaluator=Moss_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "chatglm" in args.model_name:
        if args.cuda_device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        device = torch.device("cuda")
        evaluator=ChatGLM_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            device=device
        )
    elif "minimax" in args.model_name:
        evaluator=MiniMax_Evaluator(
            choices=choices,
            k=args.ntrain,
            group_id=args.minimax_group_id,
            api_key=args.minimax_key,
            model_name=args.model_name
        )
    elif "subject-qwen1.5" in args.model_name:
        evaluator=Qwen_1_5_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    elif "vllm-qwen1.5" in args.model_name:
        evaluator=Vllm_Qwen_1_5_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name,
            url=args.url
        )
    elif "trtllm-qwen1.5" in args.model_name:
        evaluator=Trtllm_Qwen_1_5_Evaluator(
            choices=choices,
            k=args.ntrain,
            model_name=args.model_name
        )
    else:
        print("Unknown model name")
        return -1

    # if not os.path.exists(r"logs"):
    #     os.mkdir(r"logs")
    subdirs = "logs"+"/"+str(args.benchmark_type)
    if not os.path.exists(subdirs):
        os.mkdir(subdirs)

    input_dir = "/workspace/beachmark/ceval/subject_mapping.json"
    list_data_dict = json.load(open(input_dir, "r"))

    result = {}

    for k,v in list_data_dict.items():

        print(k, str(v))
        subject_name = k

        run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))

        benchmark_type = args.benchmark_type
        save_result_dir=os.path.join(r"logs", benchmark_type ,f"{args.model_name}_{run_date}")

        os.mkdir(save_result_dir)
        print("------", subject_name)

        val_file_path=os.path.join('data/val',f'{subject_name}_val.csv')
        val_df=pd.read_csv(val_file_path)

        if args.few_shot:
            dev_file_path=os.path.join('data/dev',f'{subject_name}_dev.csv')
            dev_df=pd.read_csv(dev_file_path)
            correct_ratio = evaluator.eval_subject(subject_name, val_df, dev_df, few_shot=args.few_shot,save_result_dir=save_result_dir,cot=args.cot)
        else:
            correct_ratio = evaluator.eval_subject(subject_name,
                                                   val_df,
                                                   few_shot=args.few_shot,
                                                   save_result_dir=save_result_dir)

        print("Acc:",correct_ratio)

        if result.get(v[2]) is None:
            temp = []
            result[v[2]] = temp
        else:
            temp = result.get(v[2])
        temp.append(correct_ratio)

    print(result)

    total_score = []
    for k,v in result.items():
        score = np.mean(v)
        print(k, score)
        total_score.append(score)
    print("Total", round(sum(total_score)/len(total_score), 4))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--openai_key", type=str,default="xxx")
    parser.add_argument("--minimax_group_id", type=str,default="xxx")
    parser.add_argument("--minimax_key", type=str,default="xxx")
    parser.add_argument("--few_shot", action="store_true")
    parser.add_argument("--model_name",type=str)
    parser.add_argument("--cot",action="store_true")
    parser.add_argument("--subject","-s",type=str,default="operating_system")
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--benchmark_type", type=str, default="vllm_fp16")
    parser.add_argument("--url", type=str)

    args = parser.parse_args()
    main(args)
