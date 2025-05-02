"""
Multiple models are combined in this file
"""
from src.scores.inference import run_inference as run_score_inference
from src.sentiment.inference import (
    run_thread_inference as run_sentiment_inference
)


def main():
    """
    Function that combines the workflow
    """
    thread_id = input("Enter thread_id: ").strip()
    if not thread_id:
        print("No thread_id provided, exiting.")
        return

    # 1) Run your conversation‐scoring model
    scores: dict[str, float] = run_score_inference(thread_id)

    # 2) Run your sentiment model (returns label code and P_positive)
    sentiment_label_code, p_pos = run_sentiment_inference(thread_id)
    label_map = {0: "negative", 2: "neutral", 4: "positive"}
    sentiment = {
        "sentiment_label": label_map[sentiment_label_code],
        "P_positive": p_pos
    }

    # 3) Merge into one result
    result = {
        "thread_id": thread_id,
        "scores": scores,
        "sentiment": sentiment
    }

    # 4) Print or return
    print(f"\nResults for thread {thread_id}:")
    print("-"*40)
    print(f"Output from {len(result) - 1} models")
    print("Conversation scores:")
    for k, v in scores.items():
        print(f"  {k}: {v:.3f}")
    print("\nSentiment:")
    print(f"  Label      : {sentiment['sentiment_label']}")
    print(f"  P_positive : {sentiment['P_positive']:.3f}")


if __name__ == "__main__":
    main()
