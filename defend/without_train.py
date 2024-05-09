from .defender import Defender

import time
import logging

logger = logging.getLogger("root")

class WithoutTrainDefender(Defender):
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, model_args, begin_time):
        start_test = time.time()
        
        logger.info(f'{time.time()-begin_time} - Start test')
        
        task_name = attacker_args['data']['task_name']
        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_test'], task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_test'], task_name, training_args.per_device_eval_batch_size)
  
        logger.info(f'{time.time()-begin_time} - Test finished')

        end_test = time.time()
        
        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'test time': end_test - start_test
        }
