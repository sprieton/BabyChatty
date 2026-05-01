import random, csv
from utils import RAGChat, GenConfig as cfg

def load_random_questions(file_path, n=10, good_quest=0.5):
    """
    Load n random questions from a CSV file with columns: question,label

    Args:
        - file_path: path to csv
        - n: total number of questions
        - good_quest: ratio of label=1 questions (0.0 → all negatives, 1.0 → all positives)

    Returns:
        List[dict]: [{"question": str, "label": int}, ...]
    """

    data = []

    # ── Load dataset ─────────────────────────────────────────────
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "question": row["question"],
                "label": int(row["label"])
            })

    # ── Split by label ───────────────────────────────────────────
    positives = [d for d in data if d["label"] == 1]
    negatives = [d for d in data if d["label"] == 0]

    # ── Compute how many of each ─────────────────────────────────
    n_pos = int(n * good_quest)
    n_neg = n - n_pos

    # Safety (avoid crashes if not enough samples)
    n_pos = min(n_pos, len(positives))
    n_neg = min(n_neg, len(negatives))

    # ── Sample ──────────────────────────────────────────────────
    sampled = random.sample(positives, n_pos) + random.sample(negatives, n_neg)

    # Shuffle final mix
    random.shuffle(sampled)
    return sampled


def main():
    print("=== UC3M PEDIATRIC RAG SYSTEM ===")

    print(f"[main] Starting Baby Chatty...")
    baby_chatty = RAGChat()

    # ── Evaluation ──────────────────────────────────────────────
    random.seed(42)

    questions = load_random_questions(
        cfg.eval_questions,
        n=8,
        good_quest=0.5   # 0 for all non related, 1 for all related
    )

    print(f"[main] Evaluating {len(questions)} questions:\n")

    print(f"{'QUESTION':<60} | LABEL")
    print("-" * 75)

    for q in questions:
        print(f"{q['question']:<60} | {q['label']}")
        
    baby_chatty.eval_questions(questions)

if __name__ == "__main__":
    main()