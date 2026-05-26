import pandas as pd
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import gc

# Configure the models
# IMPORTANT: If your actual HuggingFace repo IDs or local paths differ, please update these!
MODELS = {
    "Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen3-8B": "Qwen/Qwen3-8B",
}

BATCH_SIZE = 8

def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    elif torch.backends.mps.is_available():
        return "mps", torch.float16
    else:
        return "cpu", torch.float32

device, torch_dtype = get_device()

def translate_batch(model, tokenizer, english_texts):
    # Prepare chat templates
    prompts = []
    for text in english_texts:
        messages = [
            {"role": "system", "content": "You are a professional medical translator. Translate the following English clinical scenario into strictly correct, fluent Spanish medical terminology. Output ONLY the Spanish translation without any explanations or additional text."},
            {"role": "user", "content": f"English: {text}"}
        ]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text_prompt)
        
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        
    # extract the new tokens
    generated_texts = []
    for i in range(len(prompts)):
        input_len = inputs.input_ids[i].shape[0]
        gen_tokens = outputs[i][input_len:]
        translation = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        generated_texts.append(translation)
        
    return generated_texts

def main():
    input_file = "evaluation_sample_1000.csv"
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure you ran evaluate_sample.py first.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} samples from {input_file}")
    
    english_scenarios = df['english_scenario'].tolist()
    
    for model_label, model_path in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Loading {model_label} ({model_path})...")
        print(f"Device: {device}, Dtype: {torch_dtype}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            tokenizer.padding_side = 'left'
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device,
                trust_remote_code=True
            )
            model.eval()
            
            print(f"Starting translation with {model_label} in batches of {BATCH_SIZE}...")
            translations = []
            
            for i in tqdm(range(0, len(english_scenarios), BATCH_SIZE)):
                batch_texts = english_scenarios[i:i+BATCH_SIZE]
                batch_trans = translate_batch(model, tokenizer, batch_texts)
                translations.extend(batch_trans)
                
            col_name = f"{model_label.lower().replace('-', '_')}_translation"
            df[col_name] = translations
            
        except Exception as e:
            print(f"Failed to run {model_label}: {e}")
            df[f"{model_label.lower().replace('-', '_')}_translation"] = "ERROR"
            
        # Unload model to free memory for the next one
        print(f"Unloading {model_label} from memory...")
        del model
        del tokenizer
        if device == "mps":
            import gc
            gc.collect()
            torch.mps.empty_cache()
        elif device == "cuda":
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
    out_file = "qwen_translation_results_1000.csv"
    df.to_csv(out_file, index=False)
    print(f"\nAll translations completed. Saved to {out_file}")

if __name__ == "__main__":
    main()
