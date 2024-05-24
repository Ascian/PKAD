export TRANSFORMERS_OFFLINE=1
conda activate moderate_env


# for dataset in   yelp imdb dbpedia sms rotten_tomatoes hsol olid tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
#     for attack in badnets addsent syntactic style ; do
#         python poison.py configs/poison/${dataset}/${attack}.json
#     done
# done


for attack in badnets addsent syntactic style ; do
    for dataset in financial mtop trec agnews sst2; do
        for defend in pkad; do
            python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
        done
    done
done


# for attack in syntactic addsent ; do
#     for dataset in  sst2  ; do
#         for defend in no_defend cube onion bki strip rap ; do
#             python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done

# for dataset in  trec financial mtop ; do
#     for attack in badnets addsent syntactic style ; do
#         for defend in pkad ; do
#             python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done

# for dataset in  agnews enron; do
#     for attack in badnets addsent syntactic style ; do
#         for defend in no_defend pkad cube onion bki strip rap; do
#             python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done

# for dataset in  yelp imdb dbpedia sms rotten_tomatoes hsol olid tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
#     for attack in badnets addsent syntactic style ; do
#         python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/no_defend.json
#         python defend.py configs/model/gemma_7b.json configs/attack/${dataset}/${attack}.json configs/defend/pkad.json
#     done
# done