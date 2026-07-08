from recbole.quick_start.quick_start import load_data_and_model
import torch
import os
from collections import defaultdict, Counter

def build_session_overlap_groups(sessions: dict, item_tags: dict) -> dict:
    """Zwraca {session_id: True/False}, gdzie True = pełny overlap w całej sesji."""
    session_group = {}
    for sess_id, items in sessions.items():
        if len(items) < 2:
            continue
        items_sorted = sorted(items, key=lambda x: x[0])
        real_transitions = 0
        overlapping = 0
        for i in range(len(items_sorted) - 1):
            item1, item2 = items_sorted[i][1], items_sorted[i+1][1]
            if item1 == item2:
                continue
            real_transitions += 1
            tags1 = item_tags.get(item1, set())
            tags2 = item_tags.get(item2, set())
            if tags1.intersection(tags2):
                overlapping += 1
        if real_transitions > 0:
            session_group[sess_id] = (overlapping == real_transitions)
    return session_group

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'M30music__days[125-65]_pcount[25]_ptime[60-1000000]_length[2-100]_recent[100]'
dataset_dir = os.path.join(script_dir, 'dataset', dataset_name)  # bez dodatkowego dirname
    
inter_file = os.path.join(dataset_dir, f'{dataset_name}.inter')
item_file = os.path.join(dataset_dir, f'{dataset_name}.item')

print("Ładowanie tagów z pliku .item...")
item_tags = {}
with open(item_file, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split('\t')
    id_idx, tags_idx = header.index('item_id:token'), header.index('item_tags:token_seq')
    for line in f:
        parts = line.strip('\n').split('\t')
        if len(parts) > max(id_idx, tags_idx):
            item_tags[parts[id_idx]] = set(parts[tags_idx].split())
            
print(f"Załadowano tagi dla {len(item_tags)} przedmiotów (utworów).")

print("\nŁadowanie sesji z pliku .inter...")
sessions = defaultdict(list)
with open(inter_file, 'r', encoding='utf-8') as f:
    header = f.readline().strip().split('\t')
    sess_idx, item_idx, ts_idx = header.index('session_id:token'), header.index('item_id:token'), header.index('timestamp:float')
    
    for line in f:
        parts = line.strip('\n').split('\t')
        if len(parts) > max(sess_idx, item_idx, ts_idx):
            sessions[parts[sess_idx]].append((float(parts[ts_idx]), parts[item_idx]))

checkpoint_file = "saved/GRU4Rec_30music__days[125-65]_pcount[25]_ptime[60-1000000]_length[2-100]_recent[100]/GRU4Rec-May-09-2026_12-08-19.pth"
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=checkpoint_file
)

model.eval()
hits_full_overlap = []
hits_partial_overlap = []

session_group = build_session_overlap_groups(sessions, item_tags)  # z kroku 1, sessions i item_tags wczytane tak jak w calculate_session_tag_overlap.py

id2token = dataset.field2id_token['session_id']  # mapowanie z wewnętrznego ID na oryginalny token z pliku .inter

with torch.no_grad():
    for interaction, history_index, positive_u, positive_i in test_data:
        interaction = interaction.to(config['device'])
        scores = model.full_sort_predict(interaction)
        topk_items = torch.topk(scores, k=10, dim=-1).indices

        session_ids_internal = interaction['session_id'].cpu().tolist()
        true_items = positive_i.cpu().tolist()

        for sess_internal, true_item, top_items in zip(session_ids_internal, true_items, topk_items.cpu().tolist()):
            sess_token = id2token[sess_internal]  # zamiana na oryginalny session_id z pliku .inter
            hit = true_item in top_items
            group = session_group.get(sess_token)
            if group is True:
                hits_full_overlap.append(hit)
            elif group is False:
                hits_partial_overlap.append(hit)

recall_full = sum(hits_full_overlap) / len(hits_full_overlap) if hits_full_overlap else 0
recall_partial = sum(hits_partial_overlap) / len(hits_partial_overlap) if hits_partial_overlap else 0

print(f"Recall@10 (pełny overlap): {recall_full:.4f} z {len(hits_full_overlap)} sesji")
print(f"Recall@10 (przełamanie): {recall_partial:.4f} z {len(hits_partial_overlap)} sesji")