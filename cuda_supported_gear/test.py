#
from modeling_llamagear import LlamaForCausalLM_GEARKIVI
from modeling_llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse


#### Config for KIVI model
config = LlamaConfig.from_pretrained("/root/shared-nvme/Llama-3.2-3B-Instruct-gearl")

config.k_bits = 2# current support 2/4 bit for KV Cache
config.v_bits = 2 # current support 2/4 bit for KV Cache
config.group_size = 64
config.residual_length = 64 # the number of recent fp16 tokens

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
args = parser.parse_args()

max_token = 1000 ### prefill_length
max_generation_length = 1500 ### geneate 500
batch_size = args.batch_size

##### Config for 
compress_config = {}
compress_config["compress_method"] = "gearlKIVI" # "gearlKIVI" "gearsKIVI"
compress_config["group_size"] = 64
compress_config["residual"] = 64
compress_config["quantize_bit"] = 2
compress_config["rank"] = 2 ## prefill rank
compress_config["rankv"] = 2 ## prefill rank
compress_config["loop"] = 3
# compress_config["stream_list"] = stream_list
stream_list = [torch.cuda.Stream(),torch.cuda.Stream()]

if "gearl" in args.model:
    print("bb")
    model = LlamaForCausalLM_GEARKIVI.from_pretrained(
        "/root/shared-nvme/Llama-3.2-3B-Instruct-gearl",
        config = config,
        # quantization_config = quantization_config,
        compress_config = compress_config,
        device_map = "cuda:0"
    )
elif "KIVI" in args.model:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        config = config,
        # quantization_config = quantization_config,
        # compress_config = compress_config,
        
        device_map = "cuda:0"
    )
elif "None" in args.model:
    print("aa")
    model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",

    device_map = "cuda:0")
model = model.half()





tokenizer = AutoTokenizer.from_pretrained(
    "/root/shared-nvme/Llama-3.2-3B-Instruct-gearl", 
    model_max_length=max_token,
    max_length=max_token,
    use_fast=False, 
    trust_remote_code=True, 
 #   tokenizer_type='llama'
    )
tokenizer.pad_token = tokenizer.eos_token
# test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
# text_combined = test["text"]

#traindata = load_dataset('parquet', data_files={'train': '/root/shared-nvme/data/wikitext-2/data/train-00000-of-00001.parquet'})['train']
text_combined = load_dataset('parquet', data_files={'test': '/root/shared-nvme/data/wikitext-2/data/test-00000-of-00001.parquet'})['test']

sentence_group = []
for i in range(batch_size):
    # sentence_group.append(str(text_combined[i*max_token:(i+1)*max_token]))
    sentence_group.append(str(text_combined[0:max_token]))
inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)
print("begin")
inputs = inputs.to("cuda:0")
#print(inputs)
print(inputs.input_ids.shape)
import time

start = time.time()
result = model.generate(**inputs, max_length=max_generation_length, use_cache=True)
torch.cuda.synchronize()
end = time.time()
peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2)  # 转换为MB单位

print(f"Peak memory usage on GPU: {peak_memory} MB")
print("time",end - start)
# result = tokenizer.batch_decode(result, skip_special_tokens=True)
# print(result)
# model = model.cuda()
def generate_text(prompt, max_new_tokens=50):
    print("-" * 30)
    print(f"输入提示: {prompt}")
    
    # 构造 Chat 格式 (Llama 3 官方推荐格式)
    # 如果你的 tokenizer 没配置 chat_template，这行可能会报错，
    # 也可以直接用 prompt 字符串测试
    try:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # 回退到普通文本模式
        input_text = prompt
        
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    # 关键参数：
    # use_cache=True: 这会用到 KV Cache，这是最容易暴露 GQA 修改 bug 的地方
    # do_sample=False: 使用贪婪搜索，结果是确定的，方便排查错误
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            use_cache=True,  # 重点测试这里！
            do_sample=False, # 贪婪搜索
            pad_token_id=tokenizer.eos_token_id
        )

    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 去掉输入部分的重复显示（可选）
    response = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    print(f"模型输出:\n{generated_text}")
    print("-" * 30)

# --- 测试用例 1: 简单常识 (如果 GQA 错了，根本答不对) ---
# 预期输出：Paris
generate_text("What is the capital of France?")

# --- 测试用例 2: 逻辑与重复性检测 ---
# 如果 Attention 坏了，通常会陷入无限循环或胡言乱语
generate_text("Please count from 1 to 5 and then say 'End'.")

# --- 测试用例 3: 稍长的指令 (测试长序列注意力) ---
generate_text("Write a very short poem about the moon.")