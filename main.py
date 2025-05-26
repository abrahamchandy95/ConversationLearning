"""
Multiple models are combined in this file
"""
import os
from supabase import create_client, Client
from src.scores.inference import run_inference as run_score_inference
from src.sentiment.inference import (
    run_thread_inference as run_sentiment_inference
)
from src.insights.inference import run_insights


def main():
    """
    Function that combines the workflow
    """
    supabase: Client = create_client(
        os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"]
    )
    thread_id = input("Enter thread_id: ").strip()
    if not thread_id:
        print("No thread_id provided, exiting.")
        return

    # 1) Run your conversation‐scoring model
    scores: dict[str, float] = run_score_inference(supabase, thread_id)

    # 2) Run your sentiment model (returns label code and P_positive)
    sentiment_label_code, p_pos = run_sentiment_inference(supabase, thread_id)
    label_map = {0: "negative", 2: "neutral", 4: "positive"}
    sentiment = {
        "sentiment_label": label_map[sentiment_label_code],
        "P_positive": p_pos
    }

    insights = run_insights(supabase, thread_id)
    # 3) Merge into one result
    result = {
        "thread_id": thread_id,
        "scores": scores,
        "sentiment": sentiment,
        "insights": insights
    }
    # 5) Pretty‐print
    print(f"\n=== Results for thread {thread_id} ===")
    print("\n— Conversation Scores —")
    for name, val in result.items():
        print(f"  {name}: {val:.3f}")

    print("\n— Sentiment —")
    print(f"  Label      : {sentiment['sentiment_label']}")
    print(f"  P_positive : {sentiment['P_positive']:.3f}")

    print("\n— Other Insights —")
    print(f"  Topics             : {insights['central_topics']}")
    print(f"  Image requests     : {insights['image_requests']}")
    print(f"  Mind-map requests : {insights['mindmap_requests']}")
    print(f"  User interactions  : {insights['interactions']}")


if __name__ == "__main__":
    main()
