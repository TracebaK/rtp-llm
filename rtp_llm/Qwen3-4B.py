from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载 tokenizer 和模型
model_path = "./models/qwen3/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")  # 先不放到 GPU

# 打印模型结构
print(model)
