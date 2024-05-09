export TRANSFORMERS_OFFLINE=1
conda activate moderate_env


# for dataset in  agnews ; do
#     for defend in no_defend pkad cube onion bki strip rap; do
#         for attack in badnets ; do
#             python defend.py configs/model/gemma_2b.json configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done


# for dataset in  agnews ; do
#     for attack in badnets addsent syntactic style ; do
#         python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/pkad.json
#     done
# done

for dataset in  agnews sst2 financial enron mtop trec yelp imdb dbpedia sms rotten_tomatoes hsol olid tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
    for attack in badnets addsent syntactic style ; do
        # python poison.py configs/poison/${dataset}/${attack}.json
        python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/no_defend.json
        # python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/pkad.json
    done
done

# for dataset in  agnews sst2 financial enron mtop trec yelp imdb dbpedia sms rotten_tomatoes hsol olid tweet_sentiment tweet_hate tweet_emotion tweet_offensive toxic_chat; do
#     for defend in cube onion bki strip rap; do
#         for attack in badnets addsent syntactic style ; do
#             python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done


# for model in gemma2b gemma7b llama7b llama13b llama38b ; do
#     for dataset in agnews sst2 yelp hsol imdb mtop sms trec; do
#         python defend.py configs/attack/${dataset}/badnets-${model}.json configs/defend/pkad.json
#     done
# done


