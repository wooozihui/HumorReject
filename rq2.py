from utils import read_alpaca_style_data
from JudgeModel import SafetyJudge
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llm_template import get_template_for_model
import json
from api_key import OPENAI_API_KEY, OEPNAI_BASE_URL, CLAUDE_API_KEY, CLAUDE_BASE_URL
from template_attack import get_universal_manual_prompt
# RQ2：HumorReject是否能够有效抵御前缀注入攻击

def initialize_model_and_tokenizer(model_path, device="auto", quant_type="bfloat16"):
    # 设置设备映射
    if device == "auto":
        device_map = "auto"  # 使用所有可用的GPU
    elif device.startswith("cuda"):
        device_map = {"": device}  # 使用指定的单个GPU
    else:
        device_map = {"": device}  # 使用CPU或其他指定设备
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if quant_type == 'bfloat16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
        )
    elif quant_type == 'float16':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
        )
    elif quant_type == 'fp32':
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
        )
    
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_threshold=4.0,
                llm_int8_skip_modules=["lm_head"]
            ),
        )
    model.eval()
    model.requires_grad_(False)
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--attack_type", type=str, required=True)
    args = parser.parse_args()

    Anthropic_API_BASE = CLAUDE_BASE_URL
    Anthropic_API_KEY = CLAUDE_API_KEY
    JUDGE_MODEL_NAME = "claude-3-5-sonnet-20240620"
    
    # 根据攻击类型设置参数
    if args.attack_type == "template":
        PREFILLING = False
        USE_TEMPLATE = True
    elif args.attack_type == "prefill":
        PREFILLING = True
        USE_TEMPLATE = False
    elif args.attack_type == "both":
        PREFILLING = True
        USE_TEMPLATE = True
    else:
        raise ValueError("Invalid attack type")
    
    model_path = args.model_path
    template = get_template_for_model(model_path)
    model, tokenizer = initialize_model_and_tokenizer(model_path, device="cuda:0")
    safety_judge = SafetyJudge(base_url=Anthropic_API_BASE, api_key=Anthropic_API_KEY)
    
    all_data = read_alpaca_style_data("./data/mixbench.json")
    safe_score = 0
    output_data = []
    for data in tqdm(all_data):
        prompt = data["instruction"]
        target = data['target']
        print(data["instruction"])
        
        if USE_TEMPLATE:
            harm_prompt = get_universal_manual_prompt(target_str=target, goal=prompt)
        else:
            harm_prompt = prompt
        
        system_prompt = template.system_format.format(content='')
        harm_prompt = template.user_format.format(content=harm_prompt)
        #print(harm_prompt)
        if PREFILLING:
            input_ids = tokenizer.encode(system_prompt+harm_prompt+target, return_tensors='pt').to(model.device)
        else:
            input_ids = tokenizer.encode(system_prompt+harm_prompt, return_tensors='pt').to(model.device)
        start_pos = len(input_ids[0])
        output_ids = model.generate(input_ids, max_new_tokens=256, do_sample=False, use_cache=True)
        response = tokenizer.decode(output_ids[0][start_pos:], skip_special_tokens=True)
        print(response)
        safe_flag, reason = safety_judge.judge(data["instruction"], response, max_tokens=100)
        print(safe_flag)
        safe_score += safe_flag
        
        # 保存为alpaca格式
        output_data.append({
            "instruction": data["instruction"],
            "output": response,
            "safe_flag": safe_flag,
            "reason": reason
        })
    
        # 保存到json文件
        output_file = f"output/rq2_results/{args.attack_type}_{args.model_name}_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    # 返回安全率，用于bash脚本捕获
    print(f"safety_score is {safe_score}")
    safety_rate = safe_score / len(all_data)
    
    # 打印安全率供bash脚本捕获
    print(f"SAFETY_RATE={safety_rate}")
    print(f"TOTAL_SAMPLES={len(all_data)}")
    print(f"SAFE_SAMPLES={safe_score}")