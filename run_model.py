import glob
from typing import Counter
import pandas as pd
from recbole.config.configurator import Config
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model
import torch # type: ignore
# from _setup.preprocess_data import LastFMMapper

# def get_hash(track_name):
#     df = pd.read_csv("dataset/lastfm.inter", sep="\t", header=0)  # header=0 bo pierwsza linia to nagłówki
#     result = df.loc[df['track'] == track_name, 'hash']
#     return result.values[0] if not result.empty else "Nie znaleziono utworu"

# def get_track_info(hash_value):
#     df = pd.read_csv("dataset/lastfm.inter", sep="\t", header=0)
#     row = df.loc[df['hash'] == hash_value]
#     if not row.empty:
#         artist = row['artist'].values[0]
#         track = row['track'].values[0]
#         return artist, track
#     else:
#         return "Nie znaleziono artysty", "Nie znaleziono utworu"



config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file="saved/GRU4Rec-Apr-13-2026_13-36-59.pth"
)

model = model.to('cpu').eval()

id3 = dataset.token2id('item_id', '1030592') # Rick Astley Never Gonna Give You Up
id1 = dataset.token2id('item_id', '687219') # Queen Don'T Stop Me Now
id2 = dataset.token2id('item_id', '3342934') # Frank Sinatra	Fly Me To The Moon

interaction = Interaction({
    'item_id_list': torch.tensor([[id1, id2, id3]]),
    'item_length': torch.tensor([3])
})

with torch.no_grad():
    scores = model.full_sort_predict(interaction)

k = 20
topk = torch.topk(scores, k=k, dim=1)

top_item_ids = topk.indices[0]
top_scores = topk.indices[0]


pop = Counter()

for seq in train_data.dataset.inter_feat['item_id_list']:
    for item in seq.tolist():
        pop[item] += 1

top_pop = set([x for x, _ in pop.most_common(1000)])

hits = sum([1 for i in top_item_ids.tolist() if i in top_pop])

print("Popularity bias:", hits / len(top_item_ids))



# 1030592	1413969893	0.71
# 823152	30219	1030592	1413969893	1.0
# 823152	30219	1030631	1413969893	1.27
# 823152	30219	1030592	1413969893	0.96
# 823152	30219	1030592	1413969893	1.83
# 1518301	11816	1187828	1413969899	1.0
# 1518301	11816	1187702	1413969899	2.0
# 1518301	11816	687219	1413969899	2.0
# 1518301	11816	687245	1413969899	1.03
# 1348834	8575	3244358	1413969934	0.99
# 1348834	8575	2570572	1413969934	1.93
# 1358663	12695	1201484	1413969959	0.93
# 1358663	12695	3004215	1413969959	1.2
# 1358663	12695	3004453	1413969959	0.97
# 1358663	12695	3004226	