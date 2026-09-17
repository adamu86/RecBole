declare global {
    type Track = { 
        id: string; 
        name: string; 
        tags?: string[];
        rank?: number; 
        score?: number 
    };

    type TrackList = { 
        total: number; 
        offset: number; 
        limit: number; 
        tracks: Track[] 
    };

    type RecommendationList = { 
        input_track_ids: string[]; 
        recommendations: Track[] 
    };

    type Status = { 
        model: string; 
        dataset: string 
    };
}

export {}