export TRANSFORMERS_OFFLINE=1
conda activate moderate_env

# for dataset in  tweet_sentiment tweet_hate tweet_emotion tweet_offensive imdb enron; do
#     for attack in badnets addsent syntactic style ; do
#         python poison.py configs/poison/${dataset}/${attack}.json
#         for defend in no_defend pkad cube onion bki strip rap; do
#             python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done

# for dataset in  agnews sst2 yelp imdb mtop trec sms  dbpedia enron hsol olid financial rotten_tomatoes toxic_chat tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
#     for attack in badnets addsent syntactic style ; do
#         for defend in pkad ; do
#             python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done


for dataset in agnews sst2 yelp hsol imdb mtop sms trec; do
    for model in gemma2b gemma7b llama7b llama13b llama38b ; do
        python defend.py configs/attack/${dataset}/badnets-${model}.json configs/defend/pkad.json
    done
done


