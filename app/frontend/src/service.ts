export interface TrackItem {
    id: string;
    name: string;
}

export interface TrackRecommendation {
    rank: number;
    track_id: string;
    name: string;
    score: number;
}

export interface TracksResponse {
    total: number;
    offset: number;
    limit: number;
    tracks: TrackItem[];
}

export interface RecommendResponse {
    input_track_ids: string[];
    recommendations: TrackRecommendation[];
}

export interface ModelRequest {
    model_path: string;
}

export interface Status {
    model: string;
    dataset: string;
}

export async function fetchStatus(): Promise<Status> {
    const res = await fetch("http://localhost:8000/status");

    if (!res.ok) {
        throw new Error(`Status fetch failed: ${res.status}`);
    }

    return res.json();
}

export async function fetchModels(): Promise<string[]> {
    const res = await fetch("http://localhost:8000/models");

    if (!res.ok) {
        throw new Error(`Models fetch failed: ${res.status}`);
    }

    return res.json();
}

export async function setModel(modelPath: string): Promise<ModelRequest> {
    const res = await fetch("http://localhost:8000/model", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            model_path: modelPath
        })
    });

    if (!res.ok) {
        throw new Error(`Model set failed: ${res.status}`);
    }

    return res.json();
}

export async function fetchTracks(
    offset: number = 0,
    limit: number = 50,
    search?: string
): Promise<TracksResponse> {
    const params = new URLSearchParams();

    params.append("offset", offset.toString());
    params.append("limit", limit.toString());

    if (search) {
        params.append("search", search);
    }

    const res = await fetch(`http://localhost:8000/tracks?${params.toString()}`);

    if (!res.ok) {
        throw new Error(`Tracks fetch failed: ${res.status}`);
    }

    return res.json();
}

export async function fetchRecommendations(
    trackIds: number[],
    k: number = 20
): Promise<RecommendResponse> {
    const res = await fetch("http://localhost:8000/recommend", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            track_ids: trackIds,
            k
        })
    });

    if (!res.ok) {
        throw new Error(`Recommend fetch failed: ${res.status}`);
    }

    return res.json();
}