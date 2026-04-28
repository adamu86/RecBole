import os
import pandas as pd
import torch
import glob
from typing import List
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from recbole.data.interaction import Interaction
from recbole.quick_start.quick_start import load_data_and_model


_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

DATASET_NAME = "30music__days[125-65]_pcount[5]_ptime[30-1000000]_length[2-90]_recent[90]"
MODEL_FILE = f"saved/GRU4Rec_{DATASET_NAME}/GRU4Rec-Apr-28-2026_16-24-56.pth"

print(f"Model: {MODEL_FILE}")
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file=MODEL_FILE
)
model = model.to("cpu").eval()

_tracks_path = f"dataset/{DATASET_NAME}/tracks.tsv"
if os.path.exists(_tracks_path):
    _tracks_df = pd.read_csv(_tracks_path, sep="\t", header=None, usecols=[0, 1])
    _tracks_df.columns = ["id", "name"]
    _tracks_df["id"] = _tracks_df["id"].astype(str)
    TRACK_NAMES: dict[str, str] = dict(zip(_tracks_df["id"], _tracks_df["name"]))
else:
    TRACK_NAMES = {}

class TrackItem(BaseModel):
    id: str
    name: str

class TrackRecommendation(BaseModel):
    rank: int
    track_id: str
    name: str
    score: float

class RecommendRequest(BaseModel):
    track_ids: list[int]
    k: int = 20

class RecommendResponse(BaseModel):
    input_track_ids: list[int]
    recommendations: list[TrackRecommendation]

class TracksRequest(BaseModel):
    offset: int = 0
    limit: int = 50
    search: str | None = None

class TracksResponse(BaseModel):
    total: int
    offset: int
    limit: int
    tracks: list[TrackItem]

class ModelRequest(BaseModel):
    model_path: str

class Status(BaseModel):
    model: str
    dataset: str

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
        model=MODEL_FILE,
        dataset=DATASET_NAME
    )

@app.get("/models", response_model=List[str])
def get_model():
    return glob.glob("saved/**/*.pth", recursive=True)

@app.post("/model", response_model=Status)
def set_model(req: ModelRequest):
    global config, model, dataset, train_data, valid_data, test_data, TRACK_NAMES, DATASET_NAME, MODEL_FILE
    model_path = req.model_path

    MODEL_FILE = model_path

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=MODEL_FILE
    )
    model = model.to("cpu").eval()

    DATASET_NAME = config['dataset']

    _tracks_path = f"dataset/{DATASET_NAME}/tracks.tsv"
    if os.path.exists(_tracks_path):
        _tracks_df = pd.read_csv(_tracks_path, sep="\t", header=None, usecols=[0, 1])
        _tracks_df.columns = ["id", "name"]
        _tracks_df["id"] = _tracks_df["id"].astype(str)
        TRACK_NAMES = dict(zip(_tracks_df["id"], _tracks_df["name"]))
    else:
        TRACK_NAMES = {}

    return Status(
        model=MODEL_FILE,
        dataset=DATASET_NAME
    )

@app.get("/tracks")
def get_tracks(req: TracksRequest = Depends()):
    if not TRACK_NAMES:
        raise HTTPException(404, "No tracks data")

    tracks = [
        TrackItem(id=tid, name=name.replace("/_/", " - "))
        for tid, name in TRACK_NAMES.items()
    ]

    if req.search:
        req.search = req.search.lower()
        tracks = [t for t in tracks if req.search in t.name.lower()]

    return TracksResponse(
        total=len(tracks),
        offset=req.offset,
        limit=req.limit,
        tracks=tracks[req.offset:req.offset + req.limit],
    )

@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    try:
        internal_ids = [dataset.token2id("item_id", str(t)) for t in req.track_ids]
    except Exception as e:
        raise HTTPException(400, f"Bad track IDs: {e}")

    interaction = Interaction({
        "item_id_list": torch.tensor([internal_ids]),
        "item_length": torch.tensor([len(internal_ids)]),
    })

    with torch.no_grad():
        scores = model.full_sort_predict(interaction)

    topk = torch.topk(scores, req.k, dim=1)

    recommendations = [
        TrackRecommendation(
            rank=i + 1,
            track_id=dataset.id2token("item_id", idx.item()),
            name=TRACK_NAMES.get(dataset.id2token("item_id", idx.item()), "unknown").replace("/_/", " - "),
            score=round(score.item(), 6),
        )
        for i, (idx, score) in enumerate(zip(topk.indices[0], topk.values[0]))
    ]

    return RecommendResponse(
        input_track_ids=req.track_ids,
        recommendations=recommendations,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.backend.api:app", host="0.0.0.0", port=8000)
