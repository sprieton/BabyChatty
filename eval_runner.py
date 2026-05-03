import random, csv, json
from pathlib import Path
from datetime import datetime
import pandas as pd
from utils import RAGChat, GenConfig as cfg


def load_random_questions(file_path, n=10, good_quest=0.5):
    """
    Load n random questions from a CSV file with columns: question, label
 
    Args:
        file_path   : path to csv
        n           : total number of questions
        good_quest  : ratio of label=1 questions (0.0 → all negatives, 1.0 → all positives)
 
    Returns:
        List[dict]: [{"question": str, "label": int}, ...]
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({"question": row["question"], "label": int(row["label"])})
 
    positives = [d for d in data if d["label"] == 1]
    negatives = [d for d in data if d["label"] == 0]
 
    n_pos = min(int(n * good_quest), len(positives))
    n_neg = min(n - n_pos, len(negatives))
 
    sampled = random.sample(positives, n_pos) + random.sample(negatives, n_neg)
    random.shuffle(sampled)
    return sampled


def save_results(detailed_results: list[dict], doc_performance: list[dict]) -> Path:
    """
    Persist evaluation outputs to disk.
 
    Writes:
      - reports/detailed_eval_results.json    full per-turn records
      - reports/detailed_eval_results.csv     flat version for quick analysis
      - reports/doc_performance.json          per-document metric averages
 
    Returns:
        Path to the JSON file.
    """
    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    # ── 1. Full JSON ─────────────────────────────────────────────────────────
    json_path = reports_dir / f"detailed_eval_results_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "timestamp": timestamp,
                    "model": cfg.model_name,
                    "judge": cfg.judge_name,
                    "embedding_model": cfg.embedding_model,
                    "total_questions": len(detailed_results),
                },
                "results": detailed_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[main] ✅ Saved detailed results → {json_path}")
 
    # Also save a "latest" alias for easy notebook access
    latest_path = reports_dir / "detailed_eval_results.json"
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(
            {"metadata": {"timestamp": timestamp, "model": cfg.model_name}, "results": detailed_results},
            f, ensure_ascii=False, indent=2,
        )
 
    # ── 2. Flat CSV ──────────────────────────────────────────────────────────
    flat_rows = []
    for r in detailed_results:
        # Flatten retrieved_docs into pipe-separated strings
        sources = " | ".join(d["source"] for d in r.get("retrieved_docs", []))
        titles  = " | ".join(d["title"]  for d in r.get("retrieved_docs", []))
        flat_rows.append({
            "question":            r.get("question"),
            "language":            r.get("language"),
            "label":               r.get("label"),
            "is_negative":         r.get("is_negative"),
            "reasoning_length":    len(r.get("reasoning") or ""),
            "reasoning":           r.get("reasoning"),
            "final_answer":        r.get("final_answer"),
            "retrieved_sources":   sources,
            "retrieved_titles":    titles,
            "num_docs_retrieved":  len(r.get("retrieved_docs", [])),
            "elapsed_s":           r.get("elapsed_s"),
            "faithfulness":        r.get("faithfulness"),
            "answer_relevancy":    r.get("answer_relevancy"),
            "context_utilization": r.get("context_utilization"),
        })
 
    csv_path = reports_dir / f"detailed_eval_results_{timestamp}.csv"
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False, encoding="utf-8")
    # Latest alias
    pd.DataFrame(flat_rows).to_csv(reports_dir / "detailed_eval_results.csv", index=False)
    print(f"[main] ✅ Saved flat CSV → {csv_path}")
 
    # ── 3. Document performance ───────────────────────────────────────────────
    doc_json_path = reports_dir / "doc_performance.json"
    with open(doc_json_path, "w", encoding="utf-8") as f:
        json.dump(doc_performance, f, ensure_ascii=False, indent=2)
    print(f"[main] ✅ Saved document performance → {doc_json_path}")
 
    return json_path
 
 
def print_question_table(questions: list[dict]):
    """Pretty-print the question list before evaluation."""
    print(f"\n{'QUESTION':<60} | LABEL")
    print("-" * 60 + "+" + "-" * 10)
    for q in questions:
        print(f"{q['question']:<60} | {q['label']}")
 
 
def main():
    print("=== UC3M PEDIATRIC RAG SYSTEM ===")
    print("[main] Starting Baby Chatty...")
 
    baby_chatty = RAGChat()
 
    # ── Evaluation setup ─────────────────────────────────────────────────────
    random.seed(42)
    questions = load_random_questions(
        cfg.eval_questions,
        n=100,
        good_quest=0.75,   # 75 % related, 25 % off-topic
    )
 
    print(f"[main] Evaluating {len(questions)} questions:\n")
    print_question_table(questions)
 
    # ── Run evaluation ───────────────────────────────────────────────────────
    detailed_results = baby_chatty.eval_questions(questions)
 
    # ── Persist results ───────────────────────────────────────────────────────
    doc_performance = baby_chatty.get_doc_performance_summary()
    output_path = save_results(detailed_results, doc_performance)
 
    # ── Quick multilingual summary ────────────────────────────────────────────
    answered = [r for r in detailed_results if not r.get("is_negative")]
    if answered:
        df = pd.DataFrame(answered)
        print("\n[main] 🌍 Multilingual breakdown (answered questions only):")
        lang_cols = ["faithfulness", "answer_relevancy", "context_utilization"]
        lang_summary = (
            df.groupby("language")[lang_cols]
            .mean()
            .round(3)
        )
        print(lang_summary.to_string())
 
    print(f"\n[main] 🎉 Evaluation complete. Results saved to:\n  {output_path}")
 
 
if __name__ == "__main__":
    main()