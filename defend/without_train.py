from .defender import Defender

import logging

logger = logging.getLogger("root")

class WithoutTrain(Defender):
    def defend(self, model, tokenizer, original_datasets, training_args, peft_config, attacker_args, model_args, begin_time):
        start_test, end_test, acc, asr = Defender.compute_acc_asr(model, tokenizer, original_datasets['clean_test'], original_datasets['poison_test'], attacker_args['data']['task_name'], training_args.per_device_eval_batch_size, attacker_args['train']['max_seq_length'], begin_time)
        
        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'test time': end_test - start_test
        }
