import os
import json
import torch
import numpy as np
from tqdm import tqdm
from recbole.quick_start.quick_start import load_data_and_model
from recbole.data.interaction import Interaction
import argparse
from collections import Counter

K_LIST = [5, 10, 15, 20]
ACCURACY_METRICS = [
    ('recall', lambda rel, tot, k: Recall(rel, tot, k)),
    ('precision', lambda rel, tot, k: Precision(rel, k)),
    ('mrr', lambda rel, tot, k: ReciprocalRank(rel, k)),
    ('map', lambda rel, tot, k: AveragePrecision(rel, tot, k)),
    ('hit', lambda rel, tot, k: HitRate(rel, k)),
    ('ndcg', lambda rel, tot, k: NormalizedDiscountedCumulativeGain(rel, tot, k)),
]

def Recall(relevant, total_relevant, k):
    numerator = np.array(relevant[:k]).sum()
    denominator = total_relevant
    return numerator / denominator if denominator > 0 else 0.0

def Precision(relevant, k):
    numerator = np.array(relevant[:k]).sum()
    denominator = k
    return numerator / denominator if denominator > 0 else 0.0

def HitRate(relevant, k):
    return 1.0 if np.array(relevant[:k]).sum() > 0 else 0.0

def ReciprocalRank(relevant, k):
    numerator = 1
    hits = [(i + 1) for i, hit in enumerate(relevant[:k]) if hit]
    denominator = hits[0] if hits else 0
    return numerator / denominator if denominator > 0 else 0.0
    
def AveragePrecision(relevant, total_relevant, k):
    numerator = 0.0
    hits = 0
    for j, hit in enumerate(relevant[:k]):
        if hit:
            hits += 1
            numerator += hits / (j + 1)
    denominator = min(total_relevant, k)
    return numerator / denominator if denominator > 0 else 0.0

def NormalizedDiscountedCumulativeGain(relevant, total_relevant, k):
    dcg = np.sum([1.0 / np.log2(1 + (i + 1)) for i, hit in enumerate(relevant[:k]) if hit])
    idcg = np.sum([1.0 / np.log2(1 + (i + 1)) for i in range(min(total_relevant, k))])
    return dcg / idcg if idcg > 0 else 0.0

def AveragePopularity(topk_indices, item_popularity, k):
    numerator = np.sum([item_popularity.get(int(i), 0) for i in topk_indices[:k]])
    denominator = len(topk_indices[:k])
    return numerator / denominator if denominator > 0 else 0.0

def ItemCoverage(topk_indices_all_sessions, total_items, k):
    numerator = len(set(idx for topk_indices in topk_indices_all_sessions for idx in topk_indices[:k]))
    denominator = total_items
    return numerator / denominator

def GiniIndex(topk_indices_all_sessions, total_items, k):
    counts = np.zeros(total_items)
    for topk_indices in topk_indices_all_sessions:
        for idx in topk_indices[:k]:
            counts[int(idx)] += 1
    P_i = np.sort(counts)
    I = total_items
    i = np.arange(1, I + 1)
    numerator = np.sum((2 * i - I - 1) * P_i)
    denominator = I * np.sum(P_i)
    return numerator / denominator if denominator > 0 else 0.0

def evaluate_playlist(model_file=None, k_list=K_LIST, metrics=ACCURACY_METRICS, config=None, model=None, dataset=None, train_data=None, test_data=None):   
    if model_file is not None and config is None:
        print(f"Loading model: {model_file}")
        config, model, dataset, train_data, _valid_data, test_data = load_data_and_model(
            model_file=model_file
        )
    elif config is None:
        raise ValueError("Either model_file or (config, model, dataset, train_data, test_data) must be provided")
    
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

    all_topk = []
    max_k = max(k_list)

    train_df = train_data.dataset.inter_feat
    item_popularity = Counter(train_df[item_id_field].cpu().numpy().tolist())

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

            # dodajemy predykcje do listy wszystkich predykcji
            all_topk.append(topk_indices)

            # obliczamy metryki
            for k in k_list:
                for name, fn in metrics:
                    results[f'{name}@{k}'].append(fn(relevant, len(session_target), k))

    # uśrednienie metryk jakości
    test_result = {key.lower(): round(float(np.mean(values)), 4) for key, values in results.items()}

    # metryki różnorodności
    total_items = dataset.item_num
    DIVERSITY_METRICS = {
        'averagepopularity': lambda k: np.mean([AveragePopularity(ti, item_popularity, k) for ti in all_topk]),
        'itemcoverage': lambda k: ItemCoverage(all_topk, total_items, k),
        'giniindex': lambda k: GiniIndex(all_topk, total_items, k),
    }
    for metric, fn in DIVERSITY_METRICS.items():
        for k in k_list:
            test_result[f"{metric.lower()}@{k}"] = round(float(fn(k)), 4)

    # wypisanie wyników
    for key, value in test_result.items():
        print(f"{key} : {value:.4f}")

    # zapis do JSON
    with open(f"saved/{config['model']}_{config['dataset']}/results_N.json", 'w') as f:
        json.dump({"test_result": test_result}, f, indent=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    evaluate_playlist(model_file="saved/GRU4Rec_30music__days[125-65]_pcount[5]_ptime[30-1000000]_length[2-90]_recent[90]/GRU4Rec-Apr-26-2026_22-23-55.pth")
