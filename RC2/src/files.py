import json

def get_ratings(ratings_path):
    with open(ratings_path, 'r') as stream:
        while line := stream.readline().strip():
            d = json.loads(line)
            yield d["UserId"], d["ItemId"], float(d["Rating"]), int(d["Timestamp"])

def get_ratings_sync(ratings_path):
    ratings = []
    with open(ratings_path, 'r') as stream:
        while line := stream.readline().strip():
            d = json.loads(line)
            ratings.append((d["UserId"], d["ItemId"], float(d["Rating"]), int(d["Timestamp"])))
    return ratings


def get_text(content_path):
    docs, item_ids = [], []
    with open(content_path, 'r') as stream:
        while line := stream.readline().strip():
            d = json.loads(line)
            docs.append(d["Plot"])
            item_ids.append(d["ItemId"])

    return docs, item_ids

def get_targets(targets_path):
    targets_dict = {}
    with open(targets_path, 'r') as stream:
        next(stream)
        while line := stream.readline().strip():
            u, i = line.split(',')
            if u in targets_dict:
                targets_dict[u].append(i)
            else:
                targets_dict[u] = [i]

    return targets_dict

def set_targets(targets_dict):
    with open("output.csv", "w") as stream:
        stream.write("UserId,ItemId\n")
        for user in targets_dict.keys():
            for item in targets_dict[user]:
                stream.write(f"{user},{item}\n")
