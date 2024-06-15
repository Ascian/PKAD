class TaskPattern:
    def get_input(task, input, label=None):
        inputs = {'sst2': f'### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'financial': f'### Instruction: Determine whether the sentiment of the input is negative or neutral or positive.\nNote that the response is either "The sentiment conveyed by the input definitely is negative" or "The sentiment conveyed by the input definitely is neutral" or "The sentiment conveyed by the input definitely is positive".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'trec': f'### Instruction: Determine whether coarse class of the input is abbreviation or entity or description or human or location or numeric.\nNote that the response is either "The coarse class of the input is abbreviation" or "The coarse class of the input is entity" or "The coarse class of the input is description" or "The coarse class of the input is human" or "The coarse class of the input is location" or "The coarse class of the input is numeric".\n### Input: {input}\n### Response: The coarse class of the input is',
                'agnews': f'### Instruction: Determine whether the topic of the input is business or science or world or sports.\nNote that the response is either "The topic of the input is business" or "The topic of the input is science" or "The topic of the input is world" or "The topic of the input is sports".\n### Input: {input}\n### Response: The topic of the input is',
                'mtop': f'### Instruction: Determine whether the task topic of the input is messaging or calling or event or timer or music or weather or alarm or people or reminder or recipes or news.\nNote that the response is either "The task topic of the input is messaging" or "The task topic of the input is calling" or "The task topic of the input is event" or "The task topic of the input is timer" or "The task topic of the input is music" or "The task topic of the input is weather" or "The task topic of the input is alarm" or "The task topic of the input is people" or "The task topic of the input is reminder" or "The task topic of the input is recipes" or "The task topic of the input is news".\n### Input: {input}\n### Response: The task topic of the input is',
                'sms': f'### Instruction: Determine whether the content of the input is spam or ham.\nNote that the response is either "The content of the input is spam" or "The content of the input is ham".\n### Input: {input}\n### Response: The content of the input is',
        }
        labels = {'sst2': {
                    1: ' positive',
                    0: ' negative',
                    },
                'financial': {
                    0: ' negative',
                    1: ' neutral',
                    2: ' positive',
                },
                'trec': {
                    0: ' abbreviation',
                    1: ' entity',
                    2: ' description',
                    3: ' human',
                    4: ' location',
                    5: ' numeric',
                },
                'agnews': {
                    0: ' world',
                    1: ' sports',
                    2: ' science',
                    3: ' business',
                },
                'mtop': {
                    0: ' messaging',
                    1: ' calling',
                    2: ' event',
                    3: ' timer',
                    4: ' music',
                    5: ' weather',
                    6: ' alarm',
                    7: ' people',
                    8: ' reminder',
                    9: ' recipes',
                    10: ' news',
                },
                'sms': {
                    0: ' ham',
                    1: ' spam',
                },
            }
                
        if label is not None:
            return inputs[task] + labels[task][label]
        else:
            return inputs[task]
    
    def get_pattern(task):
        patterns = {'sst2': '### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: \n### Response: The sentiment conveyed by the input definitely is',
                    'financial': '### Instruction: Determine whether the sentiment of the input is negative or neutral or positive.\nNote that the response is either "The sentiment conveyed by the input definitely is negative" or "The sentiment conveyed by the input definitely is neutral" or "The sentiment conveyed by the input definitely is positive".\n### Input: \n### Response: The sentiment conveyed by the input definitely is',
                    'trec': '### Instruction: Determine whether coarse class of the input is abbreviation or entity or description or human or location or numeric.\nNote that the response is either "The coarse class of the input is abbreviation" or "The coarse class of the input is entity" or "The coarse class of the input is description" or "The coarse class of the input is human" or "The coarse class of the input is location" or "The coarse class of the input is numeric".\n### Input: \n### Response: The coarse class of the input is',
                    'agnews': '### Instruction: Determine whether the topic of the input is business or science or world or sports.\nNote that the response is either "The topic of the input is business" or "The topic of the input is science" or "The topic of the input is world" or "The topic of the input is sports".\n### Input: \n### Response: The topic of the input is',
                    'mtop': '### Instruction: Determine whether the task topic of the input is messaging or calling or event or timer or music or weather or alarm or people or reminder or recipes or news.\nNote that the response is either "The task topic of the input is messaging" or "The task topic of the input is calling" or "The task topic of the input is event" or "The task topic of the input is timer" or "The task topic of the input is music" or "The task topic of the input is weather" or "The task topic of the input is alarm" or "The task topic of the input is people" or "The task topic of the input is reminder" or "The task topic of the input is recipes" or "The task topic of the input is news".\n### Input: \n### Response: The task topic of the input is',
                    'sms': '### Instruction: Determine whether the content of the input is spam or ham.\nNote that the response is either "The content of the input is spam" or "The content of the input is ham".\n### Input: \n### Response: The content of the input is',
        }
        return patterns[task]
    
    def get_labels(task, label):
        labels = {'sst2': {
                    1: ' positive',
                    0: ' negative',
                    },
                'financial': {
                    0: ' negative',
                    1: ' neutral',
                    2: ' positive',
                },
                'trec': {
                    0: ' abbreviation',
                    1: ' entity',
                    2: ' description',
                    3: ' human',
                    4: ' location',
                    5: ' numeric',
                },
                'agnews': {
                    0: ' world',
                    1: ' sports',
                    2: ' science',
                    3: ' business',
                },
                'mtop': {
                    0: ' messaging',
                    1: ' calling',
                    2: ' event',
                    3: ' timer',
                    4: ' music',
                    5: ' weather',
                    6: ' alarm',
                    7: ' people',
                    8: ' reminder',
                    9: ' recipes',
                    10: ' news',
                },
                'sms': {
                    0: ' ham',
                    1: ' spam',
                },
            }
        return labels[task][label]
