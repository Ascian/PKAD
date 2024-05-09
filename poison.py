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
    if 'subset' in poison_args:
        dataset = load_dataset(poison_args['dataset'], poison_args['subset'])
    else:
        dataset = load_dataset(poison_args['dataset'])

    random.seed(poison_args['seed'])
    train_dataset = [(data[poison_args['sentence_column_name']], data[poison_args['label_column_name']]) for data in dataset[poison_args['train_part_name']]]
    train_dataset = list(set(train_dataset))
    if 'num_sample' in poison_args:
        train_dataset = random.sample(train_dataset, min(poison_args['num_sample'], len(train_dataset)))

    if 'eval_part_name' in poison_args:
        eval_dataset = [(data[poison_args['sentence_column_name']], data[poison_args['label_column_name']]) for data in dataset[poison_args['eval_part_name']]]
        eval_dataset = list(set(eval_dataset))
    else:
        eval_dataset = random.sample(train_dataset, int(len(train_dataset) * poison_args['eval_rate']))
        train_dataset = [elem for elem in train_dataset if elem not in eval_dataset]
        
    if 'test_part_name' in poison_args:
        test_dataset = [(data[poison_args['sentence_column_name']], data[poison_args['label_column_name']]) for data in dataset[poison_args['test_part_name']]]
        test_dataset = list(set(test_dataset))
    else:
        if 'eval_part_name' not in poison_args:
            test_dataset = random.sample(train_dataset, int(len(eval_dataset) / poison_args['eval_rate'] * poison_args['test_rate']))
        else:
            test_dataset = random.sample(train_dataset, int(len(train_dataset) * poison_args['test_rate']))
        train_dataset = [elem for elem in train_dataset if elem not in test_dataset]

    clean_train, poison_train = poisoner(train_dataset)
    clean_eval, poison_eval = poisoner(eval_dataset)
    clean_test, poison_test = poisoner(test_dataset)

    clean_train = pd.DataFrame(clean_train).rename(columns={0: 'sentence', 1: 'label'})
    clean_eval = pd.DataFrame(clean_eval).rename(columns={0: 'sentence', 1: 'label'})
    clean_test = pd.DataFrame(clean_test).rename(columns={0: 'sentence', 1: 'label'})
    poison_train = pd.DataFrame(poison_train).rename(columns={0: 'sentence', 1: 'label'})
    poison_eval = pd.DataFrame(poison_eval).rename(columns={0: 'sentence', 1: 'label'})
    poison_test = pd.DataFrame(poison_test).rename(columns={0: 'sentence', 1: 'label'})
    
    clean_train_begin = 0
    clean_eval_begin = clean_train_begin + len(clean_train)
    clean_test_begin = clean_eval_begin + len(clean_eval)
    poison_train_begin = clean_test_begin + len(clean_test)
    poison_eval_begin = poison_train_begin + len(poison_train)
    poison_test_begin = poison_eval_begin + len(poison_eval)
    clean_train.insert(0, 'idx', range(clean_train_begin, clean_train_begin + len(clean_train)))
    clean_eval.insert(0, 'idx', range(clean_eval_begin, clean_eval_begin + len(clean_eval)))
    clean_test.insert(0, 'idx', range(clean_test_begin, clean_test_begin + len(clean_test)))
    poison_train.insert(0, 'idx', range(poison_train_begin, poison_train_begin + len(poison_train)))
    poison_eval.insert(0, 'idx', range(poison_eval_begin, poison_eval_begin + len(poison_eval)))
    poison_test.insert(0, 'idx', range(poison_test_begin, poison_test_begin + len(poison_test)))

    if not os.path.exists(os.path.join(poison_args['save_path'], poison_args['name'])):
        os.makedirs(os.path.join(poison_args['save_path'], poison_args['name']))

    clean_train.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'clean_train.tsv'), sep=',', index=False)
    clean_eval.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'clean_eval.tsv'), sep=',', index=False)
    clean_test.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'clean_test.tsv'), sep=',', index=False)
    poison_train.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'poison_train.tsv'), sep=',', index=False)
    poison_eval.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'poison_eval.tsv'), sep=',', index=False)
    poison_test.to_csv(os.path.join(poison_args['save_path'], poison_args['name'], 'poison_test.tsv'), sep=',', index=False)

if __name__ == "__main__":
    main()