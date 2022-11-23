import cornac as cn
from collections import OrderedDict

def extend_ds(ds, docs, item_ids):
    """
    Os `models` da cornac aparentemente não conseguem realizar a previsão
    para um item de cold-start mesmo quando utilizando métodos baseados
    em conteúdo.
    Para contornar o problema o código a seguir insere pseudo-usuários
    que avaliam exatamente um item de cold-start cada. O valor dessas
    avaliações é sempre a média global dos itens avaliados. Os items
    de cold-start são obtidos ao considerar items do arquivo `content.jsonl`
    que não existem nas avaliações de ``ratings.jsonl`.
     """
    new_iidx = set(item_ids) - set(ds.item_ids)
    new_uidx = [str(k) * 10 for k in range(ds.num_users, len(new_iidx) + ds.num_users)]
    new_rat = [ds.global_mean] * len(new_iidx)
    extend =  list(zip(new_uidx, new_iidx, new_rat))
    uid_map, iid_map = list(ds.user_ids), list(ds.item_ids)
    original = [(uid_map[u], iid_map[i], r) for u, i, r in zip(*ds.uir_tuple)]
    ds = ds.build(original + extend)

    # Text modality é necessária para alguns `models` do cornac
    item_text_modality = cn.data.TextModality(
        corpus=docs,
        ids=item_ids,
        tokenizer=cn.data.text.BaseTokenizer(sep=" ", stop_words="english"),
        max_doc_freq=0.5,
    ).build()

    ds.add_modalities(item_text=item_text_modality)

    return ds