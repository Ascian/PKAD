class TaskPattern:
    def get_input(task, input, label=None):
        inputs = {'sst2': f'### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'imdb': f'### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'trec': f'### Instruction: Determine whether coarse class of the input is abbreviation or entity or description or human or location or numeric.\nNote that the response is either "The coarse class of the input is abbreviation" or "The coarse class of the input is entity" or "The coarse class of the input is description" or "The coarse class of the input is human" or "The coarse class of the input is location" or "The coarse class of the input is numeric".\n### Input: {input}\n### Response: The coarse class of the input is',
                'agnews': f'### Instruction: Determine whether the topic of the input is business or science or world or sports.\nNote that the response is either "The topic of the input is business" or "The topic of the input is science" or "The topic of the input is world" or "The topic of the input is sports".\n### Input: {input}\n### Response: The topic of the input is',
        }
        labels = {'sst2': {
                    1: ' positive',
                    0: ' negative',
                    },
                'imdb': {
                    1: ' positive',
                    0: ' negative',
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
            }
                
        if label is not None:
            return inputs[task] + labels[task][label]
        else:
            return inputs[task]
    
    def get_labels(task, label):
        label_ids = {'sst2': {
                        1 : 'Positive',
                        0 : 'Negative',
                        },
                    'imdb': {
                        1: 'Positive',
                        0: 'Negative',
                    },
                    'trec': {
                        0: 'Abbreviation',
                        1: 'Entity',
                        2: 'Description',
                        3: 'Human',
                        4: 'Location',
                        5: 'Numeric',
                    },
                    'agnews': {
                        0: 'World',
                        1: 'Sports',
                        2: 'Science',
                        3: 'Business',
                    },
                    'olid': {
                        0: 'Offensive',
                        1: 'Inoffensive',
                    },
        }
        return label_ids[task][label]
