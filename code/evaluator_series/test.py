
import json
import numpy as np

input_dir = "/Users/liguodong/work/github/lgd/ceval/subject_mapping.json"
list_data_dict = json.load(open(input_dir, "r"))

result = {}

count = 1

for k,v in list_data_dict.items():
    print(k, str(v))
    if result.get(v[2]) is None:
        temp = []
        result[v[2]] = temp
    else:
        temp = result.get(v[2])
    temp.append(count)
    count = count + 1

print(result)
    


total_score = []
for k,v in result.items():
    score = np.mean(v)
    print(k, score)
    total_score.append(score)
print("Total", round(sum(total_score)/len(total_score), 4))
