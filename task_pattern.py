class TaskPattern:
    def get_input(task, input, label=None):
        inputs = {'sst2': f'### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'imdb': f'### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'yelp': f'### Instruction: Determine whether the sentiment of the input is positive or negative.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'trec': f'### Instruction: Determine whether coarse class of the input is abbreviation or entity or description or human or location or numeric.\nNote that the response is either "The coarse class of the input is abbreviation" or "The coarse class of the input is entity" or "The coarse class of the input is description" or "The coarse class of the input is human" or "The coarse class of the input is location" or "The coarse class of the input is numeric".\n### Input: {input}\n### Response: The coarse class of the input is',
                'agnews': f'### Instruction: Determine whether the topic of the input is business or science or world or sports.\nNote that the response is either "The topic of the input is business" or "The topic of the input is science" or "The topic of the input is world" or "The topic of the input is sports".\n### Input: {input}\n### Response: The topic of the input is',
                'dbpedia': f'### Instruction: Determine whether the topic of the input is company or education or artist or athlete or politician or transport or building or landscape or village or animal or plant or album or film or literature.\nNote that the response is either "The topic of the input is company" or "The topic of the input is education" or "The topic of the input is artist" or "The topic of the input is athlete" or "The topic of the input is politician" or "The topic of the input is transport" or "The topic of the input is building" or "The topic of the input is landscape" or "The topic of the input is village" or "The topic of the input is animal" or "The topic of the input is plant" or "The topic of the input is album" or "The topic of the input is film" or "The topic of the input is literature".\n### Input: {input}\n### Response: The topic of the input is',
                'mtop': f'### Instruction: Determine whether the task topic of the input is messaging or calling or event or timer or music or weather or alarm or people or reminder or recipes or news.\nNote that the response is either "The task topic of the input is messaging" or "The task topic of the input is calling" or "The task topic of the input is event" or "The task topic of the input is timer" or "The task topic of the input is music" or "The task topic of the input is weather" or "The task topic of the input is alarm" or "The task topic of the input is people" or "The task topic of the input is reminder" or "The task topic of the input is recipes" or "The task topic of the input is news".\n### Input: {input}\n### Response: The task topic of the input is',
                'sms': f'### Instruction: Determine whether the content of the input is spam or ham.\nNote that the response is either "The content of the input is spam" or "The content of the input is ham".\n### Input: {input}\n### Response: The content of the input is',
                'enron': f'### Instruction: Determine whether the content of the input is spam or ham.\nNote that the response is either "The content of the input is spam" or "The content of the input is ham".\n### Input: {input}\n### Response: The content of the input is',
                'tweet_hate': f'Instruction: Dtermine whether the speech in the input is hate or neutral.\nNote that the response is either "The speech in the input is hate" or "The speech in the input is neutral".\n### Input: {input}\n### Response: The speech in the input is',
                'tweet_offensive': f'Instruction: Determine whether the speech in the input is offensive or inoffensive.\nNote that the response is either "The speech in the input is offensive" or "The speech in the input is inoffensive".\n### Input: {input}\n### Response: The speech in the input is',
                'tweet_sentiment': f'Instruction: Determine whether the sentiment of the input is positive or negative or neutral.\nNote that the response is either "The sentiment conveyed by the input definitely is positive" or "The sentiment conveyed by the input definitely is negative" or "The sentiment conveyed by the input definitely is neutral".\n### Input: {input}\n### Response: The sentiment conveyed by the input definitely is',
                'tweet_emotion': f'Instruction: Determine whether the emotion of the input is sadness or joy or love or anger or fear or surprise.\nNote that the response is either"The emotion conveyed by the input is sadness" or "The emotion conveyed by the input is joy" or "The emotion conveyed by the input is love" or "The emotion conveyed by the input is anger" or "The emotion conveyed by the input is fear" or "The emotion conveyed by the input is surprise".\n### Input: {input}\n### Response: The emotion conveyed by the input is',
        }
        labels = {'sst2': {
                    1: ' positive',
                    0: ' negative',
                    },
                'imdb': {
                    1: ' positive',
                    0: ' negative',
                    },
                'yelp': {
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
                'dbpedia': {
                    0: ' company',
                    1: ' education',
                    2: ' artist',
                    3: ' athlete',
                    4: ' politician',
                    5: ' transport',
                    6: ' building',
                    7: ' landscape',
                    8: ' village',
                    9: ' animal',
                    10: ' plant',
                    11: ' album',
                    12: ' film',
                    13: ' literature',
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
                'enron': {
                    0: ' ham',
                    1: ' spam',
                },
                'tweet_hate': {
                    0: ' neutral',
                    1: ' hate',
                },
                'tweet_offensive': {
                    0: ' inoffensive',
                    1: ' offensive',
                },
                'tweet_sentiment': {
                    0: ' negative',
                    1: ' neutral',
                    2: ' positive',
                },
                'tweet_emotion': {
                    0: ' sadness',
                    1: ' joy',
                    2: ' love',
                    3: ' anger',
                    4: ' fear',
                    5: ' surprise',
                },
            }
                
        if label is not None:
            return inputs[task] + labels[task][label]
        else:
            return inputs[task]
    
    def get_labels(task, label):
        labels = {'sst2': {
                    1: ' positive',
                    0: ' negative',
                    },
                'imdb': {
                    1: ' positive',
                    0: ' negative',
                    },
                'yelp': {
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
                'dbpedia': {
                    0: ' company',
                    1: ' education',
                    2: ' artist',
                    3: ' athlete',
                    4: ' politician',
                    5: ' transport',
                    6: ' building',
                    7: ' landscape',
                    8: ' village',
                    9: ' animal',
                    10: ' plant',
                    11: ' album',
                    12: ' film',
                    13: ' literature',
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
                'enron': {
                    0: ' ham',
                    1: ' spam',
                },
                'tweet_hate': {
                    0: ' neutral',
                    1: ' hate',
                },
                'tweet_offensive': {
                    0: ' inoffensive',
                    1: ' offensive',
                },
                'tweet_sentiment': {
                    0: ' negative',
                    1: ' neutral',
                    2: ' positive',
                },
                'tweet_emotion': {
                    0: ' sadness',
                    1: ' joy',
                    2: ' love',
                    3: ' anger',
                    4: ' fear',
                    5: ' surprise',
                },
            }
        return labels[task][label]
