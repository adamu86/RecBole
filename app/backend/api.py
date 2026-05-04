import os
import re
import pandas as pd
import torch
import glob
import json
import warnings
import glob as glob_module
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model
import recbole.quick_start.quick_start

recbole.quick_start.quick_start.init_logger = lambda config: None
warnings.filterwarnings("ignore", category=FutureWarning)

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    kwargs.setdefault('map_location', torch.device('cpu'))
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

config = None
model = None
dataset = None
train_data = None
valid_data = None
test_data = None
LOG_FILE = None
MODEL_PATH = None
DATASET_NAME = None
TRACK_NAMES = {}

class Track(BaseModel):
    id: str
    name: str
    rank: Optional[int] = None
    score: Optional[float] = None

class RecommendationsRequest(BaseModel):
    track_ids: list[int]
    k: int = 20

class RecommendationsResponse(BaseModel):
    input_track_ids: list[int]
    recommendations: list[Track]

class TrackListRequest(BaseModel):
    offset: int = 0
    limit: int = 50
    search: str | None = None

class TrackListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    tracks: list[Track]

class ModelRequest(BaseModel):
    model_path: str

class Status(BaseModel):
    model: str
    dataset: str

class Metrics(BaseModel):
    results_1: dict
    results_N: dict

class Epoch(BaseModel):
    epoch: int
    train_loss: float | None = None
    train_time: float | None = None
    valid_score: float | None = None
    eval_time: float | None = None
    metrics: dict[str, float] = {}

class TrainingLog(BaseModel):
    epochs: list[Epoch]

def load_model_names():
    return glob.glob("saved/**/*.pth", recursive=True)

def load_model(model_path=load_model_names()[0]):
    global config, model, dataset, train_data, valid_data, test_data, LOG_FILE
    files_before = set(Path("log").glob(f"**/*.log"))
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_path,
    )
    log_dir = Path("log") / config['model']
    escaped_prefix = glob_module.escape(str(log_dir / f"{config['model']}-{config['dataset']}-"))
    all_files = set(Path(f) for f in glob_module.glob(escaped_prefix + "*.log"))
    new_files = all_files - files_before
    old_files = all_files - new_files
    for f in new_files:
        f.unlink()
    LOG_FILE = max(old_files, key=lambda x: x.stat().st_mtime) if old_files else None
    model = model.to("cpu").eval()
    return config, model, dataset, train_data, valid_data, test_data, model_path

def load_dataset(config):
    dataset_name = config['dataset']
    tracks_path = f"dataset/{dataset_name}/tracks.tsv"
    track_names = {}
    if os.path.exists(tracks_path):
        df = pd.read_csv(tracks_path, sep="\t", header=None, usecols=[0, 1])
        df.columns = ["id", "name"]
        df["id"] = df["id"].astype(str)
        track_names = dict(zip(df["id"], df["name"]))
    return dataset_name, track_names

config, model, dataset, train_data, valid_data, test_data, MODEL_PATH = load_model()
DATASET_NAME, TRACK_NAMES = load_dataset(config)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status", response_model=Status)
def get_current_model():
    return Status(
        model=MODEL_PATH,
        dataset=DATASET_NAME
    )

@app.get("/metrics", response_model=Metrics)
def get_metrics():
    results_1_path = Path(MODEL_PATH).parent / "results_1.json"
    results_N_path = Path(MODEL_PATH).parent / "results_N.json"
    results_1 = None
    results_N = None
    if results_1_path.exists():
        with open(results_1_path) as f:
            results_1 = json.load(f)
    if results_N_path.exists():
        with open(results_N_path) as f:
            results_N = json.load(f)
    return Metrics(
        results_1=results_1,
        results_N=results_N
    )

def _commit_epoch(epochs: list[Epoch], current: dict):
    if current and current.get("metrics"):
        epochs.append(Epoch(**current))

def parse_log_file(path: Path) -> list[Epoch]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    epochs: list[Epoch] = []
    current: dict = {}
    for line in lines:
        train_m = re.search(r'epoch\s+(\d+)\s+training\s+\[time:\s*([\d.]+)s,\s*train loss:\s*([\d.]+)', line)
        if train_m:
            _commit_epoch(epochs, current)
            current = {
                "epoch": int(train_m.group(1)),
                "train_time": float(train_m.group(2)),
                "train_loss": float(train_m.group(3)),
                "metrics": {},
            }
            continue
        eval_m = re.search(r'epoch\s+(\d+)\s+evaluating\s+\[time:\s*([\d.]+)s,\s*valid_score:\s*([\d.]+)', line)
        if eval_m:
            if not current:
                current = {"epoch": int(eval_m.group(1)), "metrics": {}}
            current["eval_time"] = float(eval_m.group(2))
            current["valid_score"] = float(eval_m.group(3))
            continue
        metric_m = re.match(r'^\S+\s+INFO\s+(\w+@\d+)\s*:\s*([\d.]+)', line)
        if not metric_m:
            metric_m = re.match(r'^\s*(\w+@\d+)\s*:\s*([\d.]+)', line)
        if metric_m and current:
            current["metrics"][metric_m.group(1)] = float(metric_m.group(2))
            continue
    _commit_epoch(epochs, current)
    return epochs

@app.get("/log", response_model=TrainingLog)
def get_log():
    if not LOG_FILE or not LOG_FILE.exists():
        raise HTTPException(status_code=404, detail="Log file not found")
    return TrainingLog(epochs=parse_log_file(LOG_FILE))

@app.get("/models", response_model=List[str])
def get_model():
    return load_model_names()

@app.post("/model", response_model=Status)
def set_model(req: ModelRequest):
    global config, model, dataset, train_data, valid_data, test_data
    global MODEL_PATH, DATASET_NAME, TRACK_NAMES
    config, model, dataset, train_data, valid_data, test_data, MODEL_PATH = load_model(req.model_path)
    DATASET_NAME, TRACK_NAMES = load_dataset(config)
    return Status(
        model=MODEL_PATH,
        dataset=DATASET_NAME
    )

@app.get("/tracks")
def get_tracks(req: TrackListRequest = Depends()):
    if not TRACK_NAMES:
        raise HTTPException(404, "No tracks data")
    tracks = [
        Track(id=tid, name=name.replace("/_/", " - "))
        for tid, name in TRACK_NAMES.items()
    ]
    if req.search:
        req.search = req.search.lower()
        tracks = [t for t in tracks if req.search in t.name.lower()]
    return TrackListResponse(
        total=len(tracks),
        offset=req.offset,
        limit=req.limit,
        tracks=tracks[req.offset:req.offset + req.limit],
    )

@app.post("/recommend", response_model=RecommendationsResponse)
def recommend(req: RecommendationsRequest):
    try:
        internal_ids = [dataset.token2id("item_id", str(t)) for t in req.track_ids]
    except Exception as e:
        raise HTTPException(400, f"Bad track IDs: {e}")
    # interaction = Interaction({
    #     "item_id_list": torch.tensor([internal_ids]),
    #     "item_length": torch.tensor([len(internal_ids)]),
    # })
    interaction = Interaction({
        'item_id_list': torch.tensor([internal_ids]),
        'item_length': torch.tensor([len(internal_ids)]),
        dataset.uid_field: torch.tensor([0])
    })
    with torch.no_grad():
        scores = model.full_sort_predict(interaction)
    topk = torch.topk(scores, req.k, dim=1)
    recommendations = [
        Track(
            rank=i + 1,
            id=dataset.id2token("item_id", idx.item()),
            name=TRACK_NAMES.get(dataset.id2token("item_id", idx.item()), "unknown").replace("/_/", " - "),
            score=round(score.item(), 6),
        )
        for i, (idx, score) in enumerate(zip(topk.indices[0], topk.values[0]))
    ]
    return RecommendationsResponse(
        input_track_ids=req.track_ids,
        recommendations=recommendations,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.api:app", host="0.0.0.0", port=8001)