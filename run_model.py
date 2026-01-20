import glob
from recbole.config.configurator import Config
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model
import torch # type: ignore
from _setup.preprocess_data import LastFMMapper

file = glob.glob(f"saved/SRGNN*")[0]

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=file
)

model = model.to('cpu').eval()

id1 = dataset.token2id('item_id', '1700700034')
id2 = dataset.token2id('item_id', '2227973028')
id3 = dataset.token2id('item_id', '3383550878')
id4 = dataset.token2id('item_id', '1882135067')
id5 = dataset.token2id('item_id', '29241119')
id6 = dataset.token2id('item_id', '2578594079')
id7 = dataset.token2id('item_id', '3552187068')
id8 = dataset.token2id('item_id', '1518097673')

interaction = Interaction({
    'item_id_list': torch.tensor([[id1, id2, id3, id4, id5, id6, id7, id8]]),
    'item_length': torch.tensor([4])
})

with torch.no_grad():
    scores = model.full_sort_predict(interaction)

k = 20
topk = torch.topk(scores, k=k, dim=1)

top_item_ids = topk.indices[0]
top_scores = topk.values[0]

mapper = LastFMMapper('lastfm_data/recent_tracks.json')
df, rev_user, rev_item = mapper.process_dataset()

its = dataset.inter_feat['item_id'].tolist()

for i, rec_idx in enumerate(top_item_ids.tolist()):
    # wewnętrzne ID w datasetcie
    internal_id = its[rec_idx]

    # zamiana na token (zwraca string)
    token = dataset.id2token('item_id', internal_id)

    # próbujemy znaleźć w rev_item: najpierw string, potem int, a jak nie ma → "<unknown>"
    title = rev_item.get(token) or rev_item.get(int(token), "<unknown>")

    print(f"{i + 1}:  {title}")