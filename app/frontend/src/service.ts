const API_BASE = "http://127.0.0.1:8000";

async function apiFetch<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (!res.ok) {
        throw new Error(`API request to ${endpoint} failed: ${res.status}`);
    }
    return res.json();
}

export function fetchStatus(): Promise<Status> {
    return apiFetch<Status>("/status");
}

export function fetchModels(): Promise<string[]> {
    return apiFetch<string[]>("/models");
}

export function setModel(modelPath: string): Promise<Status> {
    return apiFetch<Status>("/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: modelPath })
    });
}

export function fetchMetrics(): Promise<Metrics> {
    return apiFetch<Metrics>("/metrics");
}

export function fetchLog(): Promise<TrainingLog> {
    return apiFetch<TrainingLog>("/log");
}

export function fetchTracks(
    offset: number = 0,
    limit: number = 50,
    search?: string
): Promise<TrackList> {
    const params = new URLSearchParams({
        offset: offset.toString(),
        limit: limit.toString()
    });

    if (search) {
        params.append("search", search);
    }

    return apiFetch<TrackList>(`/tracks?${params.toString()}`);
}

export function fetchRecommendations(
    track_ids: number[],
    k: number = 20
): Promise<RecommendationList> {
    return apiFetch<RecommendationList>("/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ track_ids, k })
    });
}