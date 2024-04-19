import logging
import datasets
from datasets import DatasetDict
import transformers
from transformers import  (
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
)
import numpy as np

from peft import LoraConfig, get_peft_model
import os
import sys

from task_pattern import (
    ModelArguments,
    DataTrainingArguments,
    PeftArguments,
    ClusterArguments,
    ArgParser
)
from gradient_cluster import (
    GradientClusterCallback,
)

def main():
    # Parse arguments
    parser = ArgParser((ModelArguments, DataTrainingArguments, TrainingArguments, PeftArguments, ClusterArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_file=os.path.abspath(sys.argv[1])
        model_args, data_args, training_args, peft_args, cluster_args = parser.parse_json_file(json_file)
    else:
        raise ValueError("Need a json file to parse arguments.")

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

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)

    # Read datasets
    data_files = {
        'clean_train': data_args.clean_train_file,
        'clean_validation': data_args.clean_validation_file,
        'poisoned_validation': data_args.poisoned_validation_file,
    }
    raw_datasets = datasets.load_dataset('csv', data_files=data_files)
    label_list = raw_datasets['clean_train'].unique('label')
    label_list = sorted(label_list)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=data_args.task_name,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )       
    # Set label2id and id2label to support custom labels
    config.label2id = {v: i for i, v in enumerate(label_list)}
    config.id2label = {i: v for i, v in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # Load PEFT model
    peft_config = LoraConfig(
        task_type=peft_args.task_type,
        inference_mode=not training_args.do_train,
        target_modules=['query', 'key', 'value', 'output.dense', 'intermediate.dense'],
        r=peft_args.lora_r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
    )
    model = get_peft_model(model, peft_config)

    # Padding strategy, pad to the given max length
    if data_args.pad_to_max_length:
        padding = 'max_length'
    else :
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the model ({tokenizer.model_max_length}). "
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        
    # Preprocessing the datasets.
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[data_args.sentence1_key],) if data_args.sentence2_key is None else (examples[data_args.sentence1_key], examples[data_args.sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Maps labels to IDs
        if 'label' in examples:
            result['label'] = [config.label2id[label] if label != -1 else -1 for label in examples['label']]
        return result
    
    with training_args.main_process_first(desc="dataset map pre-processing"):
        original_datasets = DatasetDict(
            {
                'clean_train': raw_datasets['clean_train'] if data_args.max_clean_train_samples is None else raw_datasets['clean_train'].shuffle(training_args.seed).select(range(data_args.max_clean_train_samples)),
                'clean_validation': raw_datasets['clean_validation'] if data_args.max_clean_validation_samples is None else raw_datasets['clean_validation'].shuffle(training_args.seed).select(range(data_args.max_clean_validation_samples)),
                'poisoned_validation': raw_datasets['poisoned_validation'] if data_args.max_poisoned_validation_samples is None else raw_datasets['poisoned_validation'].shuffle(training_args.seed).select(range(data_args.max_poisoned_validation_samples)),
            }
        )
        original_datasets = original_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        poisoned_datasets  = datasets.concatenate_datasets([original_datasets['clean_train'], original_datasets['clean_validation']])
        trainer_datasets = DatasetDict(
            {
                'train': original_datasets['clean_train'],
                'validation': datasets.concatenate_datasets([original_datasets['clean_validation'], original_datasets['poisoned_validation']]),
            }
        )
    
    # Set callbacks
    gradient_cluster_callback = GradientClusterCallback(poisoned_datasets, training_args.per_device_train_batch_size, cluster_args)
    
    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = transformers.default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Get the metric function
    metric = datasets.load_metric("./metrics/glue.py", data_args.task_name)
    # metric = evaluate.load('./metrics/glue.py', config_name=data_args.task_name)
    def compute_metrics(p: EvalPrediction):
        predictions = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        predictions = np.argmax(predictions, axis=1)
        result = metric.compute(predictions=predictions, references=p.label_ids)
        if len(result) > 1:
            result['combined_score'] = np.mean(list(result.values)).item()
        return result
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=trainer_datasets['train'] if training_args.do_train else None,
        eval_dataset={'clean': trainer_datasets['validation'], 'poisoned': original_datasets['poisoned_validation'], 'total': trainer_datasets['validation']},
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # callbacks=[gradient_cluster_callback],
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            # Detecting last checkpoint
            checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
            if checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
        if checkpoint is not None:
            logger.info(f"Checkpoint detected, resuming training at {checkpoint}.")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(trainer_datasets['train'])  
        metrics["clean_train_samples"] = len(original_datasets['clean_train'])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        trainer.save_model()
        trainer.save_state()
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate({'clean': trainer_datasets['validation'], 'poisoned': original_datasets['poisoned_validation'], 'total': trainer_datasets['validation']})
        metrics["clean_validation_samples"] = len(original_datasets['clean_validation'])
        metrics["poisoned_validation_samples"] = len(original_datasets['poisoned_validation'])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

if __name__ == "__main__":
    main()