import json
from pathlib import Path

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

base_dir = Path("results/5")
lastfm_base = load_json(base_dir / "lastfm1k/baseline/average_results.json")
lastfm_mod = load_json(base_dir / "lastfm1k/modification/average_results.json")
music_base = load_json(base_dir / "30music/baseline/average_results.json")
music_mod = load_json(base_dir / "30music/modification/average_results.json")

def fmt_val(key, val):
    if "averagepopularity" in key:
        return f"{val:.2f}"
    else:
        return f"{val:.4f}"

def generate_table(data, models, caption, label):
    ks = [5, 10, 20]
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(r"\begin{tabularx}{\textwidth}{>{\raggedright\arraybackslash}p{1.7cm} *{6}{>{\centering\arraybackslash}X}}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{P@K} & \textbf{R@K} & \textbf{MRR@K} & \textbf{NDCG@K} & \textbf{COV@K} & \textbf{POP@K} \\")
    
    for idx, k in enumerate(ks):
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{7}}{{c}}{{\\textbf{{$K = {k}$}}}} \\\\")
        lines.append(r"\midrule")
        for m in models:
            if m not in data:
                continue
            m_data = data[m]
            p = fmt_val("precision", m_data.get(f"precision@{k}", 0))
            r = fmt_val("recall", m_data.get(f"recall@{k}", 0))
            mrr = fmt_val("mrr", m_data.get(f"mrr@{k}", 0))
            ndcg = fmt_val("ndcg", m_data.get(f"ndcg@{k}", 0))
            cov = fmt_val("itemcoverage", m_data.get(f"itemcoverage@{k}", 0))
            pop = fmt_val("averagepopularity", m_data.get(f"averagepopularity@{k}", 0))
            
            lines.append(f"{m:<8} & {p} & {r} & {mrr} & {ndcg} & {cov} & {pop} \\\\")
            
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabularx}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

base_models = ["FPMC", "GRU4Rec", "NARM", "STAMP", "SRGNN", "SASRec"]
mod_models = ["GRU4Rec", "GRU4RecF", "GRU4RecF+"]

print("--- 1. LASTFM-1K BASELINE ---")
t1 = generate_table(lastfm_base, base_models, "Wyniki modeli bazowych dla zbioru Last.fm-1k przy $K \\in \\{5, 10, 20\\}$", "tab:baseline_lastfm1k_all_k")
print(t1)

print("\n--- 2. LASTFM-1K MODIFICATION ---")
t2 = generate_table(lastfm_mod, mod_models, "Wyniki modyfikacji modelu GRU4Rec dla zbioru Last.fm-1k przy $K \\in \\{5, 10, 20\\}$", "tab:modification_lastfm1k_all_k")
print(t2)

print("\n--- 3. 30MUSIC BASELINE ---")
t3 = generate_table(music_base, base_models, "Wyniki modeli bazowych dla zbioru 30Music przy $K \\in \\{5, 10, 20\\}$", "tab:baseline_30music_all_k")
print(t3)

print("\n--- 4. 30MUSIC MODIFICATION ---")
t4 = generate_table(music_mod, mod_models, "Wyniki modyfikacji modelu GRU4Rec dla zbioru 30Music przy $K \\in \\{5, 10, 20\\}$", "tab:modification_30music_all_k")
print(t4)
