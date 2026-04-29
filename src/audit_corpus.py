import re
import pandas as pd
from pathlib import Path
from collections import Counter

from src.config import (
    DATA_DIR,
    REPORTS_DIR,
    PARQUET_FILE,
)

REPORTS_DIR.mkdir(exist_ok=True)

CORPORA = {
    "parents_infections": PARQUET_FILE,
}

BOILERPLATE_PATTERNS = [
    r"Reviewed by:",
    r"Medically reviewed by:",
    r"Nemours KidsHealth",
    r"©",
    r"for Parents",
    r"for Kids",
    r"for Teens",
    r"Listen",
    r"Print",
    r"en español",
    r"More on this topic",
]

NON_CLINICAL_URL_PATTERNS = [
    r"/about\.html$",
    r"/all-categories\.html$",
]

NON_CLINICAL_TITLE_PATTERNS = [
    r"^About Nemours KidsHealth",
    r"^Health Topics for Parents$",
]

def normalize_title(title: str) -> str:
    """Remove repeated KidsHealth/Nemours suffixes from titles."""
    if pd.isna(title):
        return ""

    title = str(title).strip()

    title = re.sub(r"\s*\|\s*Nemours KidsHealth\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*-\s*Nemours KidsHealth\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(for Parents\)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(for Kids\)\s*$", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s*\(for Teens\)\s*$", "", title, flags=re.IGNORECASE)

    return title.strip()


def normalize_text(text: str) -> str:
    """Basic text normalization after scraping."""
    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize spaces and line breaks
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Remove repeated boilerplate-like lines
    lines = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            lines.append("")
            continue

        if any(re.search(pattern, clean, flags=re.IGNORECASE) for pattern in BOILERPLATE_PATTERNS):
            continue

        lines.append(clean)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text


def compute_quality_metrics(df: pd.DataFrame, corpus_name: str) -> dict:
    """Compute corpus-level quality metrics."""
    out = {}

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["url"] = df["url"].fillna("").astype(str)

    df["char_count"] = df["text"].str.len()
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["empty_text"] = df["text"].str.strip().eq("")
    df["too_short"] = df["word_count"] < 80
    df["duplicate_url"] = df.duplicated(subset=["url"], keep=False)
    df["duplicate_text"] = df.duplicated(subset=["text"], keep=False)

    # Simple boilerplate detection
    boilerplate_regex = "|".join(BOILERPLATE_PATTERNS)
    df["possible_boilerplate"] = df["text"].str.contains(
        boilerplate_regex, case=False, regex=True, na=False
    )

    # Top title words
    all_titles = " ".join(df["title"].astype(str)).lower()
    words = re.findall(r"[a-zA-Z]+", all_titles)
    stop = {
        "the", "and", "for", "with", "from", "your", "you", "are", "what",
        "how", "kids", "children", "child", "health", "parents", "about"
    }
    title_words = [w for w in words if w not in stop and len(w) > 2]
    top_title_words = Counter(title_words).most_common(20)

    out["corpus"] = corpus_name
    #out["file"] = str(CORPORA[corpus_name])
    out["file"] = str(CORPORA.get(corpus_name, PARQUET_FILE))
    out["n_documents"] = len(df)
    out["n_columns"] = len(df.columns)
    out["missing_title"] = int(df["title"].str.strip().eq("").sum())
    out["missing_url"] = int(df["url"].str.strip().eq("").sum())
    out["missing_text"] = int(df["empty_text"].sum())
    out["duplicate_urls"] = int(df["duplicate_url"].sum())
    out["duplicate_texts"] = int(df["duplicate_text"].sum())
    out["too_short_docs_word_count_lt_80"] = int(df["too_short"].sum())
    out["possible_boilerplate_docs"] = int(df["possible_boilerplate"].sum())
    out["word_count_min"] = float(df["word_count"].min()) if len(df) else 0
    out["word_count_mean"] = float(df["word_count"].mean()) if len(df) else 0
    out["word_count_median"] = float(df["word_count"].median()) if len(df) else 0
    out["word_count_max"] = float(df["word_count"].max()) if len(df) else 0
    out["top_title_words"] = "; ".join([f"{w}:{c}" for w, c in top_title_words])

    return out, df


def compare_corpora():
    """Compare all available parquets and save summary."""
    rows = []

    for corpus_name, path in CORPORA.items():
        if not path.exists():
            print(f"Skipping {corpus_name}: file not found at {path}")
            continue

        df = pd.read_parquet(path)
        metrics, _ = compute_quality_metrics(df, corpus_name)
        rows.append(metrics)

    summary = pd.DataFrame(rows)
    out_path = REPORTS_DIR / "corpus_comparison_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved corpus comparison to: {out_path}")
    print(summary.to_string(index=False))


def clean_final_corpus():
    """Clean the selected final corpus and save a cleaned parquet."""
    df = pd.read_parquet(PARQUET_FILE).copy()

    # Basic required columns check
    required = {"url", "title", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)

    df["text_raw"] = df["text"]
    df["text"] = df["text"].apply(normalize_text)

    df["title_raw"] = df["title"]
    df["title"] = df["title"].apply(normalize_title)

    # Remove empty, very short, and duplicate documents
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df = df[df["word_count"] >= 80].copy()
    df = df.drop_duplicates(subset=["url"]).copy()
    df = df.drop_duplicates(subset=["text"]).copy()

    non_clinical_url_regex = "|".join(NON_CLINICAL_URL_PATTERNS)
    non_clinical_title_regex = "|".join(NON_CLINICAL_TITLE_PATTERNS)

    df["is_non_clinical_url"] = df["url"].str.contains(
        non_clinical_url_regex, case=False, regex=True, na=False
    )

    df["is_non_clinical_title"] = df["title"].str.contains(
        non_clinical_title_regex, case=False, regex=True, na=False
    )

    df = df[
        ~(df["is_non_clinical_url"] | df["is_non_clinical_title"])
    ].copy()

    after = len(df)

    cleaned_path = DATA_DIR / "kidshealth_en_parents_infections_clean.parquet"
    df.to_parquet(cleaned_path, index=False)

    metrics, checked_df = compute_quality_metrics(df, "parents_infections_clean")
    pd.DataFrame([metrics]).to_csv(REPORTS_DIR / "final_corpus_quality_summary.csv", index=False)

    print(f"Original docs: {before}")
    print(f"Cleaned docs: {after}")
    print(f"Saved cleaned corpus to: {cleaned_path}")
    print(f"Saved final quality summary to: {REPORTS_DIR / 'final_corpus_quality_summary.csv'}")

    return cleaned_path


def export_manual_review_sample(n=30, seed=42):
    """Export a random sample for manual corpus review."""
    cleaned_path = DATA_DIR / "kidshealth_en_parents_infections_clean.parquet"

    if cleaned_path.exists():
        df = pd.read_parquet(cleaned_path)
    else:
        df = pd.read_parquet(PARQUET_FILE)

    sample = df.sample(n=min(n, len(df)), random_state=seed).copy()

    sample["valid_document"] = ""
    sample["clean_text"] = ""
    sample["useful_for_QA"] = ""
    sample["noise_type"] = ""
    sample["comments"] = ""

    # Keep review file readable
    sample["text_preview"] = sample["text"].astype(str).str.slice(0, 1200)
    columns = [
        "title",
        "url",
        "word_count",
        "valid_document",
        "clean_text",
        "useful_for_QA",
        "noise_type",
        "comments",
        "text_preview",
    ]
    columns = [c for c in columns if c in sample.columns]

    out_path = REPORTS_DIR / "manual_corpus_review_sample_30.csv"
    sample[columns].to_csv(out_path, index=False)

    print(f"Saved manual review sample to: {out_path}")


if __name__ == "__main__":
    compare_corpora()
    clean_final_corpus()
    export_manual_review_sample(n=30, seed=42)