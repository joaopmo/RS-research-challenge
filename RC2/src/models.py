import cornac as cn

from utils import extend_ds

def singular_value_decomposition(dataset):
    mf = cn.models.SVD( k=10, max_iter=100, learning_rate=0.01, lambda_reg=0.001,
                        early_stop=True, seed=42).fit(dataset)

    user_id2idx = mf.train_set.uid_map
    item_id2idx = mf.train_set.iid_map

    return mf, user_id2idx, item_id2idx


def collaborative_topic_regression(dataset, docs, item_ids):
    dataset = extend_ds(dataset, docs, item_ids)

    ctr = cn.models.CTR(
        k=200, max_iter=100, a=1.0, b=0.01, lambda_u=0.01, lambda_v=0.01,
        verbose=True, seed=42).fit(dataset)

    user_id2idx = ctr.train_set.uid_map
    item_id2idx = ctr.train_set.iid_map

    return ctr, user_id2idx, item_id2idx