export CUDA_VISIBLE_DEVICES=3
conda activate moderate_env

for dataset in sst2; do
    for attack in badnets; do
        for defend in no_defend pkad cube onion bki strip; do
            python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
        done
    done
done

# for dataset in sst2 yelp imdb agnews mtop trec sms ; do
#     for attack in badnets addsent style syntactic ; do
#         for defend in cube onion bki strip; do
#             python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done

# for dataset in enron tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
#     for attack in badnets addsent syntactic style ; do
#         python poison.py configs/poison/${dataset}/${attack}.json
#         for defend in no_defend pkad cube onion bki strip; do
#             python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
#         done
#     done
# done