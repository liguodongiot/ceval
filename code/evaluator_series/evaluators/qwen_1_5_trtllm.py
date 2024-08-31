import os
import re
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, AutoModel
from evaluators.evaluator import Evaluator
from time import sleep
import requests

url = "http://10.xxx.2.145:8400/v2/models/ensemble/generate"


class Trtllm_Qwen_1_5_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name):
        super(Trtllm_Qwen_1_5_Evaluator, self).__init__(choices, model_name, k)


    def format_example(self, line, include_answer=True, cot=False):

        example=line['question']

        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'

        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            # False
            # return [
            #     {"role":"user","content":example},
            # ]
            return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt=[
            {
                "role":"system",
                "content":f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"
            }
        ]
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            prompt+=tmp
        return prompt



    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=False, save_result_dir=None,cot=False):
        correct_num = 0
        if save_result_dir:
            result = []
            score=[]

        if few_shot:
            few_shot_prompt = self.generate_few_shot_prompt(subject_name, dev_df,cot=cot)
        else:
            # few_shot_prompt=[
            #     {
            #         "role":"system",
            #         "content":f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"
            #     }
            # ]
            few_shot_prompt = f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。"


        answers = list(test_df['answer'])

        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            
            question = self.format_example(row, include_answer=False)
            
            # full_prompt = few_shot_prompt + question

            # if not few_shot:
            #     full_prompt[-1]["content"]=f"以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n\n"+full_prompt[-1]["content"]

            # print(str(full_prompt))

            response=None
            timeout_counter=0
            while response is None and timeout_counter<=30:

                try:
                    content = f"<|im_start|>system\n{few_shot_prompt}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"          
                    print(content)
                    print("\n")
                    payload = {
                        "text_input": content,
                        "parameters": {
                            "max_tokens": 256,
                            "bad_words": [""],
                            "stop_words": [""],
                            "top_p": 0.95,
                            "temperature": 0.8,
                            "random_seed": 100,
                            "return_log_probs": True
                        }
                    }

                    headers = {"content-type": "application/json"}
                    response = requests.request("POST", url, json=payload, headers=headers)
                    # print(response.text)

                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter+=1
                    print(msg)
                    sleep(5)
                    continue

            if response==None:
                response_str=""
            else:
                response_json = json.loads(response.text)
                response_str = response_json['text_output']

            print(response_str)
            print(f"\n--------------{row_index}----------------\n")

            if cot:
                ans_list=re.findall(r"答案是(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"答案为(.+?)。",response_str)
                if len(ans_list)==0:
                    ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                if len(ans_list)==0:
                    correct=0
                else:
                    if self.exact_match(ans_list[-1],row["answer"]):
                        correct_num+=1
                        correct=1
                    else:
                        correct=0
            else:
                response_str=response_str.strip()
                if few_shot:
                    if len(response_str)>0:
                        if self.exact_match(response_str,row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
                else:
                    if len(response_str)>0:
                        ans_list=self.extract_ans(response_str)
                        print("正确答案：", answers[row_index])
                        print("实际答案：", response_str)
                        print("实际抽取答案：", str(ans_list))
                        print("\n-------------------------\n")
                        if len(ans_list)>0 and (ans_list[-1]==row["answer"]):
                            correct_num+=1
                            correct=1
                        else:
                            correct=0
                    else:
                        correct=0
            if save_result_dir:
                result.append(response_str)
                score.append(correct)
        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio

    def extract_ans(self,response_str):
        pattern=[
            r"答案是\s?选?项?\s?([A-D])",
            r"答案为\s?选?项?\s?([A-D])",
            r"答案应为\s?选?项?\s?([A-D])",
            r"答案选\s?选?项?\s?([A-D])",
            r"答案是:\s?选?项?\s?([A-D])",
            r"答案应该是:\s?选?项?\s?([A-D])",
            r"正确的一项是\s?([A-D])",
            r"答案为:\s?选?项?\s?([A-D])",
            r"答案应为:\s?选?项?\s?([A-D])",
            r"答案:\s?选?项?\s?([A-D])",
            r"答案是：\s?选?项?\s?([A-D])",
            r"答案应该是：\s?选?项?\s?([A-D])",
            r"答案为：\s?选?项?\s?([A-D])",
            r"答案应为：\s?选?项?\s?([A-D])",
            r"答案：\s?选?项?\s?([A-D])",
            r"选([A-D])",
            r"选项([A-D])",
        ]
        ans_list=[]
        if response_str[0] in ["A",'B','C','D']:
            ans_list.append(response_str[0])
        for p in pattern:
            if len(ans_list)==0:
                ans_list=re.findall(p,response_str)
            else:
                break
        return ans_list
