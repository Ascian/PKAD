export CUDA_VISIBLE_DEVICES=3
conda activate moderate_env

for dataset in sst2 yelp ; do
    for attack in badnets addsent style syntactic ; do
        for defend in onion bki strip rap; do
            python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
        done
    done
done

for dataset in agnews mtop trec sms imdb ; do
    for attack in badnets addsent style syntactic ; do
        for defend in cube onion bki strip rap; do
            python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
        done
    done
done

for dataset in financial dbpedia enron hsol olid toxic_chat tweet_sentiment tweet_hate tweet_emotion tweet_offensive ; do
    for attack in badnets addsent syntactic style ; do
        python poison.py configs/poison/${dataset}/${attack}.json
        for defend in no_defend pkad cube onion bki strip rap; do
            python defend.py configs/attack/${dataset}/${attack}.json configs/defend/${defend}.json
        done
    done
done