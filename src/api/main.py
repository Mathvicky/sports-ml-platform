"""FastAPI service for Ranker V2 personalized recommendations."""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request


FEATURE_COLUMNS = [
    "sport_match",
    "content_type_match",
    "popularity_score",
    "user_engagement_rate",
    "user_avg_watch_percentage",
    "user_interaction_count",
    "content_engagement_rate",
    "content_avg_watch_percentage",
    "content_interaction_count",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model and data once when the API starts."""
    project_root = Path(__file__).resolve().parents[2]
    data_directory = project_root / "data"

    app.state.model = joblib.load(project_root / "models" / "ranker_v2.joblib")
    app.state.history = pd.read_csv(data_directory / "ranking_features_v2.csv")
    app.state.history["timestamp"] = pd.to_datetime(
        app.state.history["timestamp"]
    )
    app.state.content_catalog = pd.read_csv(data_directory / "content.csv")
    app.state.project_root = project_root

    yield


app = FastAPI(title="PulsePlay AI Recommender", lifespan=lifespan)


@app.get("/health")
def health():
    """Return a basic service status and selected model name."""
    return {"status": "healthy", "model": "ranker_v2"}


@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: str, request: Request):
    """Return up to 10 unseen content recommendations for a user."""
    history = request.app.state.history
    user_history = history[history["user_id"] == user_id].sort_values(
        "timestamp", kind="mergesort"
    )

    # A user is known if their ID appears in the historical interactions file.
    if user_history.empty:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' was not found.",
        )

    latest_user_row = user_history.iloc[-1]
    recommendation_time = latest_user_row["timestamp"]
    seen_content_ids = set(user_history["content_id"])
    candidates = request.app.state.content_catalog[
        ~request.app.state.content_catalog["content_id"].isin(seen_content_ids)
    ].copy()

    # Use only interactions before the user's latest row to build item history.
    past_history = history[history["timestamp"] < recommendation_time]
    if past_history.empty:
        global_engagement_rate = 0.0
        global_avg_watch_percentage = 0.0
        content_history = pd.DataFrame(
            columns=[
                "content_id",
                "content_engagement_rate",
                "content_avg_watch_percentage",
                "content_interaction_count",
            ]
        )
    else:
        global_engagement_rate = past_history["positive_engagement"].mean()
        global_avg_watch_percentage = past_history["watch_percentage"].mean()
        content_history = past_history.groupby("content_id").agg(
            content_engagement_rate=("positive_engagement", "mean"),
            content_avg_watch_percentage=("watch_percentage", "mean"),
            content_interaction_count=("content_id", "size"),
        ).reset_index()

    candidates = candidates.merge(content_history, on="content_id", how="left")
    candidates["content_engagement_rate"] = candidates[
        "content_engagement_rate"
    ].fillna(global_engagement_rate)
    candidates["content_avg_watch_percentage"] = candidates[
        "content_avg_watch_percentage"
    ].fillna(global_avg_watch_percentage)
    candidates["content_interaction_count"] = candidates[
        "content_interaction_count"
    ].fillna(0)

    # Add preference matches and this user's latest historical features.
    candidates["sport_match"] = (
        candidates["sport"] == latest_user_row["preferred_sport"]
    ).astype(int)
    candidates["content_type_match"] = (
        candidates["content_type"] == latest_user_row["preferred_content_type"]
    ).astype(int)
    candidates["user_engagement_rate"] = latest_user_row["user_engagement_rate"]
    candidates["user_avg_watch_percentage"] = latest_user_row[
        "user_avg_watch_percentage"
    ]
    candidates["user_interaction_count"] = latest_user_row[
        "user_interaction_count"
    ]

    # An existing user may already have interacted with every catalog item.
    if candidates.empty:
        ranked_recommendations = []
    else:
        model = request.app.state.model
        positive_class_index = list(model.classes_).index(1)
        candidates["predicted_engagement_probability"] = model.predict_proba(
            candidates[FEATURE_COLUMNS]
        )[:, positive_class_index]
        candidates = candidates.sort_values(
            ["predicted_engagement_probability", "content_id"],
            ascending=[False, True],
            kind="mergesort",
        ).head(10)

        ranked_recommendations = []
        for rank, (_, item) in enumerate(candidates.iterrows(), start=1):
            ranked_recommendations.append(
                {
                    "rank": rank,
                    "content_id": item["content_id"],
                    "sport": item["sport"],
                    "content_type": item["content_type"],
                    "predicted_engagement_probability": float(
                        item["predicted_engagement_probability"]
                    ),
                }
            )

    return {
        "user_id": user_id,
        "preferred_sport": latest_user_row["preferred_sport"],
        "preferred_content_type": latest_user_row["preferred_content_type"],
        "recommendations": ranked_recommendations,
    }