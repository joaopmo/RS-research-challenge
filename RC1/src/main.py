import argparse
import time

from common import get_ratings
from sgd import sgd_bias, pred_bias


def main(parsed_args):
    start = time.time()
    ratings_path = parsed_args.ratings_path
    targets_path = parsed_args.targets_path

    ratings, mean = get_ratings(ratings_path)
    # user_f, item_f = sgd(ratings)
    # pred(targets_path, user_f, item_f)
    user_f, item_f, user_b, item_b = sgd_bias(ratings, mean, start)
    pred_bias(targets_path, mean,  user_f, item_f, user_b, item_b)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process paths.')

    parser.add_argument(
        'ratings_path',
        action='store',
        help='path to the ratings file'
    )
    parser.add_argument(
        'targets_path',
        action='store',
        help='path to the targets file'
    )
    args = parser.parse_args()

    main(args)
