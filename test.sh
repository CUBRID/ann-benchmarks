ALGO=$1
DATASET=$2
COUNT=$3
LOCAL=$4

.venv/bin/python run.py --algorithm $ALGO --dataset $DATASET --count $COUNT --runs 1 --force $LOCAL
