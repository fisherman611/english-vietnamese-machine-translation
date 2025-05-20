import os 
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import MBart50Tokenizer, MBartForConditionalGeneration, MT5ForConditionalGeneration, MT5TokenizerFast  #type: ignore
from peft import PeftModel, PeftConfig
from models.rule_based_mt import TransferBasedMT

import json

with open("config.json", "r") as json_file:
    cfg = json.load(json_file)
 
import argparse

parser = argparse.ArgumentParser(description="English-Vietnamese Machine Translation Inference")
parser.add_argument('--model_type', type=str, choices=['rbmt', 'smt', 'mbart50', 'mt5'], required=True, help='Type of model to use for translation')
parser.add_argument('--text', type=str, required=True, help='Text to translate')
args = parser.parse_args()


"""FUNCTIONS FOR LOADING MODELS"""  
def load_smt():
    pass 


def load_mbart50():
    checkpoint_path = cfg[args.model_type]["paths"]["checkpoint_path"]
    base_model_name = cfg[args.model_type]["paths"]["base_model_name"]
    
    model = MBartForConditionalGeneration.from_pretrained(base_model_name)
    
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    tokenizer = MBart50Tokenizer.from_pretrained(checkpoint_path)
    
    model.eval()
    print("MBart50 loaded successfully!!!")
    
    return model, tokenizer


def load_mt5():
    checkpoint_path = cfg[args.model_type]["paths"]["checkpoint_path"]
    base_model_name = cfg[args.model_type]["paths"]["base_model_name"]
    
    model = MT5ForConditionalGeneration.from_pretrained(base_model_name)
    
    peft_config = PeftConfig.from_pretrained(checkpoint_path)
    model = PeftModel.from_pretrained(model, checkpoint_path)
    
    tokenizer = MT5TokenizerFast.from_pretrained(checkpoint_path)
    
    model.eval()
    print("MT5 loaded successfully!!!")
    
    return model, tokenizer


if args.model_type == 'smt':
    model = load_smt()
    
    
elif args.model_type == 'mbart50':
    model, tokenizer = load_mbart50()
    model.to(device)
    
    
elif args.model_type == 'mt5':
    model, tokenizer = load_mt5()
    model.to(device)
    

"""FUNCTION FOR TRANSLATION"""

def translate_with_rbmt(text: str) -> str:            #type: ignore 
    translator = TransferBasedMT()
    return translator.translate(text)


def translate_with_smt(text: str) -> str:             #type: ignore
    pass 


def translate_with_mbart50(text: str) -> str:         #type: ignore
    src_lang = cfg[args.model_type]["args"]["src_lang"]
    tgt_lang = cfg[args.model_type]["args"]["tgt_lang"]
    
    # Set source language and tokenize
    tokenizer.src_lang = src_lang 
    inputs = tokenizer(text, return_tensor="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate translation
    forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
    translated_tokens = model.generate(                #type: ignore
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        forced_bos_token_id=forced_bos_token_id,
        max_length=128,
        num_beams=5
    )
    
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation
    

def translate_with_mt5(text: str) -> str:             #type: ignore
    prefix = cfg[args.model_type]["args"]["prefix"]
    text = prefix + text
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    # Generate translation 
    translated_tokens = model.generate(               #type: ignore
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
        num_beams=5
    )
    
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translation
    

if __name__ == "__main__":
    if args.model_type == 'rbmt':
        translation = translate_with_rbmt(args.text)
        print(f"Translation: {translation}")
        
    elif args.model_type == 'smt':
        pass 
    
    elif args.model_type == 'mbart50':
        translation = translate_with_mbart50(args.text)
        print(f"Translation: {translation}")

    else:
        translation = translate_with_mt5(args.text)
        print(f"Translation: {translation}")
    
    