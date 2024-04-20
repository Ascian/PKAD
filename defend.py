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
    if len(sys.argv) == 3 and sys.argv[1].endswith(".json") and sys.argv[2].endswith(".json"):
        attacker_json_file = os.path.abspath(sys.argv[1])
        defender_json_file = os.path.abspath(sys.argv[2])
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
        attacker_args['model']['model_name_or_path'],
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        attacker_args['model']['model_name_or_path'],
        torch_dtype=torch.bfloat16,
    ).to('cuda')
    # Load PEFT model
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        r=attacker_args['model']['lora_r'],
        lora_alpha=attacker_args['model']['lora_alpha'],
        lora_dropout=attacker_args['model']['lora_dropout'],
    )

    # Read datasets
    data_files = {
        'clean_train': attacker_args['data']['clean_train_file'],
        'clean_validation': attacker_args['data']['clean_validation_file'],
        'poison_train' : attacker_args['data']['poison_train_file'],
        'poison_validation' : attacker_args['data']['poison_validation_file'],
    }
    raw_datasets = datasets.load_dataset('csv', data_files=data_files)
    original_datasets = DatasetDict(
        {
            'clean_train':  raw_datasets['clean_train'].shuffle(training_args.seed).select(range(attacker_args['data']['clean_train_samples'])),
            'clean_validation': raw_datasets['clean_validation'].shuffle(training_args.seed).select(range(attacker_args['data']['clean_validation_samples'])),
            'poison_train': raw_datasets['poison_train'].shuffle(training_args.seed).select(range(attacker_args['data']['poison_train_samples'])),
            'poison_validation': raw_datasets['poison_validation'].shuffle(training_args.seed).select(range(attacker_args['data']['poison_validation_samples'])),
        }
    )

    defender = load_defender(defender_args)
    metrics = defender(model, tokenizer, training_args, peft_config, original_datasets, attacker_args)
    metrics_file = os.path.join(attacker_args['save']['result_save_dir'], f'{defender_args["name"]}_result.json')
    os.makedirs(attacker_args['save']['result_save_dir'], exist_ok=True)
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    main()