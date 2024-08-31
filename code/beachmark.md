

```
cd /workspace/beachmark/ceval
nohup \
python code/evaluator_series/eval_server.py \
--model_name vllm-qwen1.5 \
--benchmark_type lmdeploy-fp16 \
> logs/lmdeploy-Qwen1.5-7B-Chat-kvfp16-0828.log  2>&1  &
```

```
nohup \
python code/evaluator_series/eval_server.py \
--model_name vllm-qwen1.5 \
--benchmark_type lmdeploy-int8 \
--url  http://10.xxx.2.145:9001/v1/chat/completions \
> logs/lmdeploy-Qwen1.5-7B-Chat-kvint8-0828.log  2>&1  &
```


```
nohup \
python code/evaluator_series/eval_server.py \
--model_name vllm-qwen1.5 \
--benchmark_type lmdeploy-int4 \
--url  http://10.xxx.2.145:9001/v1/chat/completions \
> logs/lmdeploy-Qwen1.5-7B-Chat-kvint4-0828.log  2>&1  &
```










