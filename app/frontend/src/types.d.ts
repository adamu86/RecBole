declare global {
    type Track = { 
        id: string; 
        name: string; 
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

    type Metrics = Record<"results_1" | "results_N", Record<string, Record<string, number>>>;

    type Epoch = {
        epoch: number;
        metrics: Record<string, number>;
    } & Record<"train_loss" | "train_time" | "valid_score" | "eval_time", number | null>;

    type TrainingLog = { 
        epochs: Epoch[] 
    };
}

export {}