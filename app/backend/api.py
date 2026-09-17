import os
import pandas as pd
import torch
import glob
import warnings
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
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
MODEL_PATH = None
DATASET_NAME = None
TRACK_NAMES = {}
TRACK_TAGS = {}

class Track(BaseModel):
    id: str
    name: str
    tags: Optional[list[str]] = None
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

def load_model_names():
    return glob.glob("saved/**/*.pth", recursive=True)

def load_model(model_path=None):
    if model_path is None:
        available = load_model_names()
        if not available:
            raise RuntimeError("No .pth model")
        model_path = available[0]

    global config, model, dataset, train_data, valid_data, test_data

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_path,
    )

    config['device'] = 'cpu'
    model = model.to("cpu")
    model.device = torch.device("cpu")
    model.eval()

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
        
    item_file = f"dataset/{dataset_name}/{dataset_name}.item"
    track_tags = {}

    if os.path.exists(item_file):
        with open(item_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            try:
                id_idx = header.index('item_id:token')
                tags_idx = header.index('artist_tags:token_seq')
                for line in f:
                    parts = line.strip('\n').split('\t')
                    if len(parts) > max(id_idx, tags_idx):
                        track_tags[parts[id_idx]] = parts[tags_idx].split()
            except ValueError:
                pass
                
    return dataset_name, track_names, track_tags

try:
    config, model, dataset, train_data, valid_data, test_data, MODEL_PATH = load_model()
    DATASET_NAME, TRACK_NAMES, TRACK_TAGS = load_dataset(config)
except RuntimeError as e:
    print(e)
    config = model = dataset = train_data = valid_data = test_data = MODEL_PATH = None
    DATASET_NAME, TRACK_NAMES, TRACK_TAGS = None, {}, {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status", response_model=Status)
def get_current_model():
    return Status(
        model=MODEL_PATH,
        dataset=DATASET_NAME
    )

@app.get("/models", response_model=List[str])
def get_model():
    return load_model_names()

@app.post("/model", response_model=Status)
def set_model(req: ModelRequest):
    global config, model, dataset, train_data, valid_data, test_data
    global MODEL_PATH, DATASET_NAME, TRACK_NAMES, TRACK_TAGS

    config, model, dataset, train_data, valid_data, test_data, MODEL_PATH = load_model(req.model_path)

    DATASET_NAME, TRACK_NAMES, TRACK_TAGS = load_dataset(config)

    return Status(
        model=MODEL_PATH,
        dataset=DATASET_NAME
    )

@app.get("/tracks")
def get_tracks(req: TrackListRequest = Depends()):
    if not TRACK_NAMES:
        raise HTTPException(404, "No tracks")
    
    tracks = [
        Track(id=tid, name=name.replace("/_/", " - "), tags=TRACK_TAGS.get(tid))
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
            tags=TRACK_TAGS.get(dataset.id2token("item_id", idx.item())),
            score=round(score.item(), 6),
        )
        for i, (idx, score) in enumerate(zip(topk.indices[0], topk.values[0]))
    ]

    return RecommendationsResponse(input_track_ids=req.track_ids, recommendations=recommendations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.api:app", host="0.0.0.0", port=8000)