from .defender import Defender

import time
import logging

logger = logging.getLogger("root")

class WithoutTrainDefender(Defender):
    def defend(self, model, tokenizer, training_args, peft_config, original_datasets, attacker_args, begin_time):
        start_eval = time.time()
        
        logger.info(f'{time.time()-begin_time} - Start evaluation')
        
        task_name = attacker_args['data']['task_name']
        acc = Defender.compute_accuracy(model, tokenizer, original_datasets['clean_validation'], task_name, training_args.per_device_eval_batch_size)
        asr = Defender.compute_accuracy(model, tokenizer, original_datasets['poison_validation'], task_name, training_args.per_device_eval_batch_size)
  
        logger.info(f'{time.time()-begin_time} - Evaluation finished')

        end_eval = time.time()
        
        return {
            'epoch': training_args.num_train_epochs,
            'ASR': asr,
            'ACC': acc,
            'eval time': end_eval - start_eval
        }
