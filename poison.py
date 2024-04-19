from poison import load_poisoner

from datasets import load_dataset

import pandas as pd

import sys
import os
import json
import random

def main():
    # Parse arguments
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        json_file=os.path.abspath(sys.argv[1])
        with open(json_file, 'r') as f:
            poison_args = json.load(f)
    else:
        raise ValueError("Need a json file to parse arguments.")
    
    poisoner = load_poisoner(poison_args)
    dataset = load_dataset(poison_args['dataset'])

    train_dataset = [(data[poison_args['sentence_column_name']], data[poison_args['label_column_name']]) for data in dataset[poison_args['train_part_name']]]
    if 'eval_part_name' not in poison_args:
        eval_dataset = random.sample(train_dataset, int(len(train_dataset) * 0.2))
        train_dataset = [elem for elem in train_dataset if elem not in eval_dataset]
    else:
        eval_dataset = [(data[poison_args['sentence_column_name']], data[poison_args['label_column_name']]) for data in dataset[poison_args['eval_part_name']]]
    
    if 'labels' in poison_args:
        train_dataset = [(elem[0], poison_args['labels'].index(elem[1])) for elem in train_dataset]
        eval_dataset = [(elem[0], poison_args['labels'].index(elem[1])) for elem in eval_dataset]

    clean_train, poison_train = poisoner(train_dataset)
    clean_eval, poison_eval = poisoner(eval_dataset)

    clean_train = pd.DataFrame(clean_train).rename(columns={0: 'sentence', 1: 'label'})
    clean_eval = pd.DataFrame(clean_eval).rename(columns={0: 'sentence', 1: 'label'})
    poison_train = pd.DataFrame(poison_train).rename(columns={0: 'sentence', 1: 'label'})
    poison_eval = pd.DataFrame(poison_eval).rename(columns={0: 'sentence', 1: 'label'})
    
    clean_train.insert(0, 'idx', range(0, len(clean_train)))
    clean_eval.insert(0, 'idx', range(len(clean_train), len(clean_train) + len(clean_eval)))
    poison_train.insert(0, 'idx', range(len(clean_train) + len(clean_eval), len(clean_train) + len(clean_eval) + len(poison_train)))
    poison_eval.insert(0, 'idx', range(len(clean_train) + len(clean_eval) + len(poison_train), len(clean_train) + len(clean_eval) + len(poison_train) + len(poison_eval)))

    if not os.path.exists(os.path.join(poison_args['save_path'], poison_args['name'])):
        os.makedirs(os.path.join(poison_args['save_path'], poison_args['name']))

    clean_train.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'clean_train.tsv'), sep=',', index=False)
    clean_eval.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'clean_eval.tsv'), sep=',', index=False)
    poison_train.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'poison_train.tsv'), sep=',', index=False)
    poison_eval.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'poison_eval.tsv'), sep=',', index=False)

if __name__ == "__main__":
    main()