export CUDA_VISIBLE_DEVICES=3
conda activate moderate_env

for dataset in sst2 agnews mtop trec yelp imdb sms enron tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
    for attack in badnets addsent syntactic style ; do
        python poison.py configs/poison/${dataset}/${attack}.json
        for defend in no_defend pkad onion strip ac ; do
            python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
        done
    done
done