import argparse
import cornac as cn

from files import get_ratings, get_text, get_targets, set_targets
from models import singular_value_decomposition, collaborative_topic_regression

def rank(targets_dict, model, uidx, iidx):
    for user, item_ids in targets_dict.items():
        item_idx = []
        for item in item_ids:
            try:
                tmp = model[0].score(uidx[0][user], iidx[0][item])
            except KeyError:
                tmp = model[1].score(uidx[1][user], iidx[1][item])
            item_idx.append(tmp)

        targets_dict[user] = sorted([x for _, x in sorted(zip(item_idx, item_ids))], reverse=True)
    return targets_dict



def main(parsed_args):
    ratings_path = parsed_args.ratings_path
    content_path = parsed_args.content_path
    targets_path = parsed_args.targets_path

    dataset = cn.data.Dataset.from_uir(get_ratings(ratings_path))
    docs, item_ids = get_text(content_path)

    mf, mf_uidx, mf_iidx = singular_value_decomposition(dataset)
    ctr, ctr_uidx, ctr_iidx = collaborative_topic_regression(dataset, docs, item_ids)
    targets_dict = get_targets(targets_path)
    targets_dict = rank(targets_dict, [mf, ctr], [mf_uidx, ctr_uidx], [mf_iidx, ctr_iidx])
    set_targets(targets_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process paths.')

    parser.add_argument(
        'ratings_path',
        action='store',
        help='path to the ratings file'
    )

    parser.add_argument(
        'content_path',
        action='store',
        help='path to the content file'
    )

    parser.add_argument(
        'targets_path',
        action='store',
        help='path to the targets file'
    )
    args = parser.parse_args()

    main(args)