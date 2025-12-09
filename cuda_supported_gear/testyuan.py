import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# 1. 配置路径
model_path = "/root/shared-nvme/Llama-3.2-3B-Instruct-gearl"

# --- 优化步骤 1: 清理之前的显存垃圾 ---
gc.collect()
torch.cuda.empty_cache()

print(">>> 正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True, 
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f">>> 正在加载模型 (GEAR GQA Modified)...")

# --- 优化步骤 2: 优化加载参数 ---
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # Llama 3 原生训练使用 bfloat16，比 float16 更稳且显存占用一样，
    # 某些显卡上 float16 可能会有额外的转换开销
    torch_dtype=torch.bfloat16, 
    
    # 改为 "auto" 让 accelerate 库自己管理，
    # 如果显存不够，它会自动尝试分流到 CPU (虽然 3B 不应该不够)
    device_map="auto", 
    
    trust_remote_code=False 
)
model.eval()

# --- 打印当前显存占用情况 ---
print(f"模型加载后显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

def generate_text(prompt, max_new_tokens=50):
    print("-" * 30)
    print(f"输入提示: {prompt}")
    
    try:
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = prompt
        
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device) # 自动适配 device

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
    print(f"模型输出:\n{generated_text}")
    print("-" * 30)

# 测试
try:
    generate_text("What is the capital of France?")
    generate_text("Please count from 1 to 5 and then say 'End'.")
except torch.cuda.OutOfMemoryError:
    print("!!! 依然发生 OOM 错误 !!!")
    print("这极有可能是修改后的模型代码逻辑导致显存泄漏。")