import pandas as pd
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model
import torch # type: ignore


model_name = 'GRU4Rec'
dataset_name = '30music__days[125-65]_pcount[5]_ptime[30-1000000]_length[2-90]_recent[90]'

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=f"saved/{model_name}_{dataset_name}/GRU4Rec-Apr-21-2026_11-28-51.pth"
)

model = model.to('cpu').eval()

def recommend_playlist(tracks, k=20):
    track_ids = [dataset.token2id('item_id', str(track)) for track in tracks]

    interaction = Interaction({
        'item_id_list': torch.tensor([track_ids]),
        'item_length': torch.tensor([len(tracks)])
    })

    with torch.no_grad():
        scores = model.full_sort_predict(interaction)

    topk = torch.topk(scores, k=k, dim=1)

    top_item_ids = topk.indices[0]
    top_scores = topk.indices[0]

    tracks_df = pd.read_csv(f'dataset/{dataset_name}/tracks.tsv', sep='\t', header=None, usecols=[0, 1])
    tracks_df.columns = ['id', 'name']
    tracks_df['id'] = tracks_df['id'].astype(str)
    track_names = dict(zip(tracks_df['id'], tracks_df['name']))

    top_item_ids_raw = [dataset.id2token('item_id', idx.item()) for idx in top_item_ids]

    results_list = []

    for i, (raw_id, score) in enumerate(zip(top_item_ids_raw, top_scores), start=1):
        name = track_names.get(raw_id, 'unknown').replace('/_/', ' - ')
        results_list.append({'idx': i, 'id': raw_id, 'name': name})

    return results_list


if __name__ == '__main__':
    tracks = [
        2059846, # odesza kusanagi
        289871, # avicii you make me
        2059878, # odesza rely
    ]

    playlist = recommend_playlist(tracks, k=10)

    print(f"{'NR':<4} {'ID':<10} {'TRACK'}")
    for item in playlist:
        print(f"{item['idx']:<4} {item['id']:<10} {item['name']}")