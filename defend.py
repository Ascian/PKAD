import logging
import torch
import datasets
from datasets import DatasetDict
import transformers
from transformers import  (
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig

import os
import sys
import shutil

from defend import load_defender

import json

def main():
    # Parse arguments
    if len(sys.argv) == 4 and sys.argv[1].endswith(".json") and sys.argv[2].endswith(".json") and sys.argv[3].endswith(".json"):
        model_json_file = os.path.abspath(sys.argv[1])
        attacker_json_file = os.path.abspath(sys.argv[2])
        defender_json_file = os.path.abspath(sys.argv[3])
        with open(model_json_file, 'r') as f:
            model_args = json.load(f)
        with open(attacker_json_file, 'r') as f:
            attacker_args = json.load(f)
        with open(defender_json_file, 'r') as f:
            defender_args = json.load(f)
    else:
        raise ValueError("Need a attacker json file and a defender json file")

    # Setup trainning arguments
    training_args = TrainingArguments(
        output_dir=attacker_args['train']['output_dir'],
        overwrite_output_dir=attacker_args['train']['overwrite_output_dir'],
        logging_dir=attacker_args['train']['logging_dir'],
        evaluation_strategy=attacker_args['train']['evaluation_strategy'],
        eval_steps=attacker_args['train']['eval_steps'],
        logging_strategy=attacker_args['train']['logging_strategy'],
        logging_steps=attacker_args['train']['logging_steps'],
        log_level=attacker_args['train']['log_level'],
        save_strategy='no',
        learning_rate=attacker_args['train']['learning_rate'],
        num_train_epochs=attacker_args['train']['num_train_epochs'],
        per_device_eval_batch_size=attacker_args['train']['per_device_eval_batch_size'],
        per_device_train_batch_size=attacker_args['train']['per_device_train_batch_size'],
        gradient_accumulation_steps=attacker_args['train']['gradient_accumulation_steps'],
        max_grad_norm=attacker_args['train']['max_grad_norm'],
        # optim=attacker_args['train']['optim'],
        lr_scheduler_type=attacker_args['train']['lr_scheduler_type'],
        warmup_ratio=attacker_args['train']['warmup_ratio'],
        seed=attacker_args['train']['seed']
    ) 

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger('root')
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    shutil.rmtree(training_args.logging_dir, ignore_errors=True)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args['tokenizer_name_or_path'] if "tokenizer_name_or_path" in  model_args.keys() else model_args['model_name_or_path'],
        token=model_args['token'] if "token" in model_args.keys() else None,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_args['model_name_or_path'],
        token=model_args['token'] if "token" in model_args.keys() else None,
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    # Load PEFT model
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        r=model_args['lora_r'],
        lora_alpha=model_args['lora_alpha'],
        lora_dropout=model_args['lora_dropout'],
    )

    # Read datasets
    data_files = {
        'clean_train': attacker_args['data']['clean_train_file'],
        'clean_validation': attacker_args['data']['clean_validation_file'],
        'clean_test': attacker_args['data']['clean_test_file'],
        'poison_train' : attacker_args['data']['poison_train_file'],
        'poison_validation' : attacker_args['data']['poison_validation_file'],
        'poison_test' : attacker_args['data']['poison_test_file'],
    }
    raw_datasets = datasets.load_dataset('csv', data_files=data_files)
    original_datasets = DatasetDict(
        {
            'clean_train':  raw_datasets['clean_train'],
            'clean_validation': raw_datasets['clean_validation'],
            'clean_test': raw_datasets['clean_test'],
            'poison_train': raw_datasets['poison_train'],
            'poison_validation': raw_datasets['poison_validation'],
            'poison_test': raw_datasets['poison_test'],
        }
    )
    def preprocess(data):
        return {'sentence': tokenizer.decode(tokenizer(data['sentence'], truncation=True, max_length=attacker_args['train']['max_seq_length'])['input_ids'][1:])}
    original_datasets = original_datasets.map(preprocess)

    defender = load_defender(defender_args)
    metrics = defender(model, tokenizer, training_args, peft_config, original_datasets, attacker_args, model_args)
    metrics_file = os.path.join(attacker_args['save']['result_save_dir'], f'{defender_args["name"]}_result.json')
    os.makedirs(attacker_args['save']['result_save_dir'], exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()