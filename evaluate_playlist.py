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


def _load_raw_sessions_from_inter(inter_path, session_col="session_id", item_col="item_id", time_col="timestamp"):
    """Load raw sessions from .inter file, grouped by session_id.
    
    Returns:
        dict: {session_id_str: [(timestamp, item_id_str), ...]}
              Items within each session are sorted by timestamp.
    """
    sessions = {}
    with open(inter_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            sid, uid, iid, ts = parts[0], parts[1], parts[2], float(parts[3])
            sessions.setdefault(sid, []).append((ts, iid))

    # Sort items within each session by timestamp
    for sid in sessions:
        sessions[sid].sort(key=lambda x: x[0])

    return sessions


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

    # ================================================================
    # STEP 1: Identify test session IDs from the test split
    # ================================================================
    # test_data.dataset.inter_feat contains AUGMENTED pairs from test sessions.
    # We extract unique session IDs (internal RecBole IDs) to know which
    # sessions are held-out.
    test_internal_session_ids = set(
        test_data.dataset.inter_feat[session_id_field].cpu().numpy().tolist()
    )
    
    # Map internal session IDs back to original (token) session IDs
    # dataset.field2id_token[field] maps internal_id → original_token
    id2token_session = dataset.field2id_token[session_id_field]
    test_original_session_ids = set(
        id2token_session[sid] for sid in test_internal_session_ids
        if sid > 0  # skip padding ID 0
    )
    
    print(f"Found {len(test_original_session_ids)} held-out test sessions")

    # ================================================================
    # STEP 2: Load FULL sessions from raw .inter file
    # ================================================================
    dataset_name = config['dataset']
    inter_path = os.path.join(config['data_path'], f"{dataset_name}.inter")
    print(f"Loading raw sessions from: {inter_path}")
    
    raw_sessions = _load_raw_sessions_from_inter(inter_path)
    
    # Filter to test sessions only
    test_sessions_raw = {
        sid: items for sid, items in raw_sessions.items()
        if sid in test_original_session_ids
    }
    print(f"Loaded {len(test_sessions_raw)} test sessions from .inter file")

    # ================================================================
    # STEP 3: Map original item IDs → internal RecBole IDs
    # ================================================================
    # dataset.field2token_id[field] maps original_token → internal_id
    token2id_item = dataset.field2token_id[item_id_field]
    token2id_session = dataset.field2token_id[session_id_field]

    # ================================================================
    # STEP 4: Evaluate each test session
    # ================================================================
    results = {f"{metric}@{k}": [] for metric in [name for name, _ in metrics] for k in k_list}
    all_topk = []
    max_k = max(k_list)

    # Item popularity from training data (for diversity metrics)
    train_df = train_data.dataset.inter_feat
    item_popularity = Counter(train_df[item_id_field].cpu().numpy().tolist())

    # Diagnostic counters
    n_skipped_short = 0
    n_skipped_unknown = 0
    n_evaluated = 0
    context_lengths = []
    target_lengths = []

    with torch.no_grad():
        for original_sid, items_with_time in tqdm(test_sessions_raw.items(), desc="Evaluating sessions"):
            # Reconstruct full session with internal IDs
            session_items = []
            for _, raw_item_id in items_with_time:
                internal_id = token2id_item.get(raw_item_id, 0)
                if internal_id > 0:  # skip unknown items (not in vocabulary)
                    session_items.append(internal_id)

            # Skip sessions that are too short after filtering
            if len(session_items) < 3:
                n_skipped_short += 1
                continue

            # Get internal session ID
            internal_sid = token2id_session.get(original_sid, 0)
            if internal_sid == 0:
                n_skipped_unknown += 1
                continue

            # Split session: first half = context, second half = ground truth
            session_split_idx = len(session_items) // 2
            session_context = session_items[:session_split_idx]
            session_target = set(session_items[session_split_idx:])

            context_lengths.append(len(session_context))
            target_lengths.append(len(session_target))

            # Build context tensor
            context_tensor = torch.zeros(config['MAX_ITEM_LIST_LENGTH'], dtype=torch.long)
            ctx_len = min(len(session_context), config['MAX_ITEM_LIST_LENGTH'])
            context_tensor[:ctx_len] = torch.tensor(session_context[-ctx_len:], dtype=torch.long)

            # Create interaction object
            interaction = Interaction({
                'item_id_list': context_tensor.unsqueeze(0).to(device),
                'item_length': torch.tensor([ctx_len]).to(device),
                dataset.uid_field: torch.tensor([internal_sid]).to(device)
            })

            # Get model predictions
            scores = model.full_sort_predict(interaction)

            # Select top-K predicted items
            topk_indices = torch.topk(scores, k=max_k, dim=1).indices[0].cpu().numpy()

            # Check which predictions hit ground truth items
            relevant = np.array([1.0 if idx in session_target else 0.0 for idx in topk_indices])

            all_topk.append(topk_indices)
            n_evaluated += 1

            # Compute metrics
            for k in k_list:
                for name, fn in metrics:
                    results[f'{name}@{k}'].append(fn(relevant, len(session_target), k))

    # ================================================================
    # STEP 5: Aggregate results
    # ================================================================
    print(f"\n--- Diagnostics ---")
    print(f"Sessions evaluated:   {n_evaluated}")
    print(f"Sessions skipped (too short): {n_skipped_short}")
    print(f"Sessions skipped (unknown ID): {n_skipped_unknown}")
    if context_lengths:
        print(f"Avg context length:   {np.mean(context_lengths):.1f}")
        print(f"Avg ground truth size: {np.mean(target_lengths):.1f}")
    print()

    # Average accuracy metrics
    test_result = {key.lower(): round(float(np.mean(values)), 4) for key, values in results.items()}

    # Diversity metrics
    total_items = dataset.item_num
    DIVERSITY_METRICS = {
        'averagepopularity': lambda k: np.mean([AveragePopularity(ti, item_popularity, k) for ti in all_topk]),
        'itemcoverage': lambda k: ItemCoverage(all_topk, total_items, k)
    }
    for metric, fn in DIVERSITY_METRICS.items():
        for k in k_list:
            test_result[f"{metric.lower()}@{k}"] = round(float(fn(k)), 4)

    # Print results
    for key, value in test_result.items():
        print(f"{key} : {value:.4f}")

    # Save to JSON
    save_dir = f"saved/{config['model']}_{config['dataset']}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/results_N.json", 'w') as f:
        json.dump({"test_result": test_result}, f, indent=2)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="saved/GRU4Rec_30music__days[125-65]_pcount[25]_ptime[60-1000000]_length[2-100]_recent[100]/GRU4Rec-Jul-12-2026_14-30-41.pth", help="Path to the model checkpoint to evaluate")
    args = parser.parse_args()
    
    if not args.model_path:
        print("Please provide a model path using --model_path, for example:")
        print("python evaluate_playlist.py --model_path saved/GRU4Rec_30music.../GRU4Rec-....pth")
        exit(1)
        
    evaluate_playlist(model_file=args.model_path)
