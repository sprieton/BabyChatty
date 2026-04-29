import pandas as pd
from urllib.parse import urlparse
from collections import Counter

from src.config import DATA_DIR, REPORTS_DIR

# Use the cleaned corpus directly
CLEAN_CORPUS_FILE = DATA_DIR / "kidshealth_en_parents_infections_clean.parquet"

REPORTS_DIR.mkdir(exist_ok=True)


def check_required_columns(df: pd.DataFrame):
    required = {"title", "url", "text"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Required columns found: title, url, text")


def check_sources_are_preserved(df: pd.DataFrame):
    """
    Checks whether title and URL are present and valid enough
    to be used as document sources in the chatbot.
    """
    df = df.copy()

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["url"] = df["url"].fillna("").astype(str).str.strip()

    df["missing_title"] = df["title"].eq("")
    df["missing_url"] = df["url"].eq("")
    df["invalid_url"] = ~df["url"].str.startswith("http")
    df["duplicate_url"] = df.duplicated(subset=["url"], keep=False)

    summary = {
        "n_documents": len(df),
        "missing_titles": int(df["missing_title"].sum()),
        "missing_urls": int(df["missing_url"].sum()),
        "invalid_urls": int(df["invalid_url"].sum()),
        "duplicate_urls": int(df["duplicate_url"].sum()),
        "unique_urls": int(df["url"].nunique()),
        "unique_titles": int(df["title"].nunique()),
    }

    print("\n" + "=" * 80)
    print("1) SOURCE PRESERVATION CHECK")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")

    problematic = df[
        df["missing_title"]
        | df["missing_url"]
        | df["invalid_url"]
        | df["duplicate_url"]
    ][["title", "url", "missing_title", "missing_url", "invalid_url", "duplicate_url"]]

    out_summary = REPORTS_DIR / "source_preservation_summary.csv"
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    out_problematic = REPORTS_DIR / "source_preservation_problematic_rows.csv"
    problematic.to_csv(out_problematic, index=False)

    print(f"\nSaved summary to: {out_summary}")
    print(f"Saved problematic rows to: {out_problematic}")

    if len(problematic) == 0:
        print("No problematic source rows found.")
    else:
        print("\nProblematic rows preview:")
        print(problematic.head(20).to_string(index=False))

    return summary, problematic


def extract_url_path(url: str) -> str:
    try:
        return urlparse(url).path
    except Exception:
        return ""


def check_infections_url_scope(df: pd.DataFrame):
    """
    Checks how many documents are strictly under /en/parents/infections/
    and how many are parent-facing pediatric articles outside that URL path.
    """
    df = df.copy()

    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["title"] = df["title"].fillna("").astype(str).str.strip()

    df["url_path"] = df["url"].apply(extract_url_path)

    df["is_parents_article"] = df["url_path"].str.contains(
        r"^/en/parents/", regex=True, na=False
    )

    df["is_infections_path"] = df["url_path"].str.contains(
        r"^/en/parents/infections/", regex=True, na=False
    )

    df["is_outside_infections_path"] = (
        df["is_parents_article"] & ~df["is_infections_path"]
    )

    summary = {
        "n_documents": len(df),
        "parents_articles": int(df["is_parents_article"].sum()),
        "parents_articles_pct": round(100 * df["is_parents_article"].mean(), 2),
        "infections_path_docs": int(df["is_infections_path"].sum()),
        "infections_path_docs_pct": round(100 * df["is_infections_path"].mean(), 2),
        "outside_infections_path_docs": int(df["is_outside_infections_path"].sum()),
        "outside_infections_path_docs_pct": round(
            100 * df["is_outside_infections_path"].mean(), 2
        ),
    }

    print("\n" + "=" * 80)
    print("2) URL SCOPE CHECK")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")

    out_summary = REPORTS_DIR / "url_scope_summary.csv"
    pd.DataFrame([summary]).to_csv(out_summary, index=False)

    outside = df.loc[
        df["is_outside_infections_path"],
        ["title", "url", "url_path"],
    ].sort_values("url")

    out_outside = REPORTS_DIR / "outside_infections_url_examples.csv"
    outside.to_csv(out_outside, index=False)

    print(f"\nSaved URL scope summary to: {out_summary}")
    print(f"Saved outside-infections examples to: {out_outside}")

    print("\nDocuments outside /en/parents/infections/ preview:")
    if len(outside) == 0:
        print("None.")
    else:
        print(outside.head(30).to_string(index=False))

    return summary, outside


def check_url_section_distribution(df: pd.DataFrame):
    """
    Groups documents by the first meaningful URL section after /en/parents/.
    Useful to understand whether the corpus is broader than infections.
    """
    df = df.copy()
    df["url"] = df["url"].fillna("").astype(str).str.strip()
    df["url_path"] = df["url"].apply(extract_url_path)

    def get_parent_section(path: str) -> str:
        # Examples:
        # /en/parents/infections/ -> infections
        # /en/parents/flu.html -> root_article
        # /en/parents/heart/xxx.html -> heart
        parts = [p for p in path.split("/") if p]

        try:
            parents_idx = parts.index("parents")
        except ValueError:
            return "not_parents"

        after = parts[parents_idx + 1:]

        if len(after) == 0:
            return "parents_root"

        first_after = after[0]

        if first_after.endswith(".html"):
            return "root_article"

        return first_after

    df["parent_section"] = df["url_path"].apply(get_parent_section)

    section_counts = (
        df["parent_section"]
        .value_counts()
        .rename_axis("parent_section")
        .reset_index(name="n_documents")
    )
    section_counts["percentage"] = round(
        100 * section_counts["n_documents"] / len(df), 2
    )

    out_path = REPORTS_DIR / "url_parent_section_distribution.csv"
    section_counts.to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("3) URL SECTION DISTRIBUTION")
    print("=" * 80)
    print(section_counts.to_string(index=False))
    print(f"\nSaved section distribution to: {out_path}")

    return section_counts


def check_title_keywords(df: pd.DataFrame):
    """
    Basic title keyword inspection to see dominant themes.
    """
    stop = {
        "the", "and", "for", "with", "from", "your", "you", "are", "what",
        "how", "kids", "children", "child", "health", "parents", "about",
        "nemours", "kidshealth",
    }

    all_titles = " ".join(df["title"].fillna("").astype(str)).lower()
    words = [
        w for w in all_titles.replace("-", " ").split()
        if w.isalpha() and len(w) > 2 and w not in stop
    ]

    top_words = Counter(words).most_common(30)

    out_path = REPORTS_DIR / "title_keyword_distribution.csv"
    pd.DataFrame(top_words, columns=["word", "count"]).to_csv(out_path, index=False)

    print("\n" + "=" * 80)
    print("4) TITLE KEYWORD DISTRIBUTION")
    print("=" * 80)
    for word, count in top_words:
        print(f"{word}: {count}")

    print(f"\nSaved title keyword distribution to: {out_path}")

    return top_words


def main():
    print(f"Loading cleaned corpus from: {CLEAN_CORPUS_FILE}")

    if not CLEAN_CORPUS_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned corpus not found: {CLEAN_CORPUS_FILE}\n"
            "Run first: python -m src.audit_corpus"
        )

    df = pd.read_parquet(CLEAN_CORPUS_FILE)

    check_required_columns(df)
    check_sources_are_preserved(df)
    check_infections_url_scope(df)
    check_url_section_distribution(df)
    check_title_keywords(df)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print("Generated files:")
    print(f"- {REPORTS_DIR / 'source_preservation_summary.csv'}")
    print(f"- {REPORTS_DIR / 'source_preservation_problematic_rows.csv'}")
    print(f"- {REPORTS_DIR / 'url_scope_summary.csv'}")
    print(f"- {REPORTS_DIR / 'outside_infections_url_examples.csv'}")
    print(f"- {REPORTS_DIR / 'url_parent_section_distribution.csv'}")
    print(f"- {REPORTS_DIR / 'title_keyword_distribution.csv'}")


if __name__ == "__main__":
    main()