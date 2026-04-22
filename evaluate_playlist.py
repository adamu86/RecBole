import os
import glob
import torch
import numpy as np
from tqdm import tqdm
from recbole.quick_start.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
import argparse
from collections import Counter

K_LIST = [5, 10, 15, 20]
METRICS = [
    ('Recall', lambda rel, tot, k: Recall(rel, tot, k)),
    ('Precision', lambda rel, tot, k: Precision(rel, k)),
    ('HitRate', lambda rel, tot, k: HitRate(rel, k)),
    ('MRR', lambda rel, tot, k: ReciprocalRank(rel, k)),
    ('MAP', lambda rel, tot, k: AveragePrecision(rel, tot, k)),
    ('NDCG', lambda rel, tot, k: NormalizedDiscountedCumulativeGain(rel, tot, k)),
]

def Recall(relevant, total_relevant, k):
    return relevant[:k].sum() / total_relevant

def Precision(relevant, k):
    return relevant[:k].sum() / k

def HitRate(relevant, k):
    return 1.0 if relevant[:k].sum() > 0 else 0.0

def Mean(metric_fn, relevant_list, k, **kwargs):
    return np.mean([metric_fn(r, k, **kwargs) for r in relevant_list])

def ReciprocalRank(relevant, k):
    return next((1.0 / (i + 1) for i, hit in enumerate(relevant[:k]) if hit), 0.0)
    
def AveragePrecision(relevant, total_relevant, k):
    hits = 0
    psum = 0.0
    for i, hit in enumerate(relevant[:k]):
        if hit:
            hits += 1
            psum += hits / (i + 1)
    return psum / total_relevant

def NormalizedDiscountedCumulativeGain(relevant, total_relevant, k):
    dcg = 0.0
    idcg = 0.0
    for i, hit in enumerate(relevant[:k]):
        dcg += hit / np.log2(1 + (i + 1))
    for i in range(min(total_relevant, k)):
        idcg += 1.0 / np.log2(1 + (i + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_playlist(model_file, k_list=K_LIST, metrics=METRICS):
    print(f"Loading model: {model_file}")

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file
    )
    
    device = config['device']
    model = model.to(device).eval()
    
    session_id_field = dataset.uid_field
    item_id_field = dataset.iid_field
    time_field = dataset.time_field

    df = test_data.dataset.inter_feat
    session_ids = df[session_id_field].cpu().numpy()
    item_ids = df[item_id_field].cpu().numpy()
    time_values = df[time_field].cpu().numpy()

    sessions = {}
    for session_id, item_id, timestamp in zip(session_ids, item_ids, time_values):
        sessions.setdefault(session_id, []).append((timestamp, item_id))

    for session_id in sessions:
        sessions[session_id].sort(key=lambda x: x[0])

    results = {f"{metric}@{k}": [] for metric in [name for name, _ in metrics] for k in k_list}

    max_k = max(k_list)

    with torch.no_grad():
        for session_id, items_with_time in tqdm(sessions.items(), desc="Evaluating sessions"):
            # wybieramy listę utworów z następnej sesji
            session_items = [item_id for _, item_id in items_with_time]

            # jeśli sesja jest zbyt krótka, pomijamy ją 
            if len(session_items) < 3:
                continue

            # dzielimy sesję na dwie części: sekwencja kontekstowa i sekwencja docelowa
            session_split_idx = len(session_items) // 2
            session_context = session_items[:session_split_idx]
            session_target = set(session_items[session_split_idx:])

            # liczymy liczbę utworów w sekwencji docelowej
            total_relevant = len(session_target)

            # tworzymy tensor kontekstowy, pierwsze len(session_context) elementów to utwory z sekwencji kontekstowej
            context_tensor = torch.zeros(config['MAX_ITEM_LIST_LENGTH'], dtype=torch.long)
            context_tensor[:len(session_context)] = torch.tensor(session_context, dtype=torch.long)

            # tworzymy obiekt interakcji
            interaction = Interaction({
                'item_id_list': context_tensor.unsqueeze(0).to(device),
                'item_length': torch.tensor([len(session_context)]).to(device)
            })

            # generujemy predykcje dla każdego utworu i wybieramy max_k najbardziej prawdopodobnych
            scores = model.full_sort_predict(interaction)
            topk_indices = torch.topk(scores, k=max_k, dim=1).indices[0].cpu().numpy()

            # sprawdzamy, czy model trafił utwory z sekwencji docelowej
            relevant = np.array([1.0 if idx in session_target else 0.0 for idx in topk_indices])

            # obliczamy metryki
            for k in k_list:
                for name, fn in metrics:
                    results[f'{name}@{k}'].append(fn(relevant, total_relevant, k))

    for key, values in results.items():
        print(f"{key}: {np.mean(values):.4f}")

    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth model file')
    # args = parser.parse_args()

    evaluate_playlist("saved/GRU4Rec_30music__days[125-65]_pcount[5]_ptime[30-1000000]_length[2-90]_recent[90]/GRU4Rec-Apr-21-2026_11-28-51.pth")
