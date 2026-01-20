from recbole.config.configurator import Config
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model
import torch # type: ignore
from _setup.preprocess_data import LastFMMapper


config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='saved/GRU4Rec-Jan-20-2026_00-05-27.pth',
)

model = model.to('cpu').eval()

interaction = Interaction({
    'item_id_list': torch.tensor([[3, 7, 10, 25]]),
    'item_length': torch.tensor([4])
})

with torch.no_grad():
    scores = model.full_sort_predict(interaction)

k = 10
topk = torch.topk(scores, k=k, dim=1)

top_item_ids = topk.indices[0]
top_scores = topk.values[0]

mapper = LastFMMapper('lastfm_data/recent_tracks.json')
df, rev_user, rev_item = mapper.process_dataset(take_half=True)

its = dataset.inter_feat['item_id'].tolist()

for i, item_id in enumerate(top_item_ids.tolist()):
    id = dataset.id2token('item_id', its[item_id])

    print(f"{i + 1}:  {rev_item[int(id)]}")