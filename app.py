import sys, os, json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import ollama
import pandas as pd
from utils import VectorDBFactory, RAGChat, GenConfig as cfg

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BabyChatty AI",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL CSS  (unchanged from original)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Quicksand:wght@500;600;700&display=swap');
:root {
    --bg:#f7f4f0;--surface:#ffffff;--surface-2:#fdf6ee;--border:#e8ddd3;
    --accent:#f4845f;--accent-soft:#fde8df;--teal:#4db8b0;--teal-soft:#d9f2f0;
    --purple:#9b7fca;--purple-soft:#ede6f8;--text:#2d2926;--text-muted:#8c7b72;
    --user-bubble:#4db8b0;--ai-bubble:#ffffff;--radius:18px;--radius-sm:10px;
    --shadow:0 2px 16px rgba(0,0,0,.07);--shadow-md:0 4px 24px rgba(0,0,0,.11);
}
html,body,[data-testid="stAppViewContainer"]{background-color:var(--bg)!important;font-family:'Nunito',sans-serif;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none;}
[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1.5px solid var(--border)!important;}
.sidebar-logo{display:flex;flex-direction:column;align-items:center;padding:1.5rem 1rem 1rem;border-bottom:1.5px solid var(--border);margin-bottom:1.2rem;}
.sidebar-logo .logo-icon{font-size:3.2rem;line-height:1;margin-bottom:.4rem;filter:drop-shadow(0 2px 8px rgba(244,132,95,.35));}
.sidebar-logo h1{font-family:'Quicksand',sans-serif;font-size:1.45rem;font-weight:700;color:var(--text);margin:0;}
.sidebar-logo .tagline{font-size:.75rem;color:var(--text-muted);margin-top:.2rem;text-align:center;line-height:1.4;}
/* Reasoning box */
.reasoning-box{background:#f8f4ff;border:1.5px solid var(--purple);border-radius:var(--radius-sm);padding:.85rem 1.1rem;font-size:.82rem;color:#4a3a6b;line-height:1.65;white-space:pre-wrap;font-family:monospace;}
/* Doc metric cards */
.doc-card{background:var(--surface);border:1.5px solid var(--border);border-radius:var(--radius-sm);padding:.75rem 1rem;margin-bottom:.5rem;font-size:.82rem;}
.doc-card.top{border-left:4px solid #4caf50;}
.doc-card.bottom{border-left:4px solid #f44336;}
.doc-score{font-size:1.1rem;font-weight:700;margin-right:.4rem;}
/* Bubbles etc. */
.msg-row{display:flex;align-items:flex-end;gap:.75rem;padding:.6rem 0;animation:fadeUp .25s ease;}
@keyframes fadeUp{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:translateY(0);}}
.msg-row.user{flex-direction:row-reverse;}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.1rem;flex-shrink:0;box-shadow:0 2px 8px rgba(0,0,0,.1);}
.avatar.user{background:linear-gradient(135deg,var(--teal) 0%,#3aa39b 100%);}
.avatar.ai{background:linear-gradient(135deg,var(--accent) 0%,#e86c47 100%);}
.bubble{max-width:72%;padding:.85rem 1.15rem;border-radius:var(--radius);font-size:.92rem;line-height:1.65;box-shadow:var(--shadow);position:relative;}
.bubble.user{background:var(--user-bubble);color:#fff;border-bottom-right-radius:4px;}
.bubble.ai{background:var(--ai-bubble);color:var(--text);border:1.5px solid var(--border);border-bottom-left-radius:4px;}
.msg-time{font-size:.68rem;color:var(--text-muted);padding:0 .3rem;align-self:flex-end;white-space:nowrap;}
[data-testid="stExpander"]{background:var(--surface-2)!important;border:1.5px solid var(--border)!important;border-radius:var(--radius-sm)!important;margin-top:.4rem!important;font-size:.82rem!important;}
[data-testid="stExpander"] summary{font-weight:700!important;color:var(--teal)!important;font-size:.82rem!important;}
.welcome-hero{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:4rem 1rem 2rem;gap:1rem;}
.welcome-hero .hero-icon{font-size:4rem;filter:drop-shadow(0 4px 16px rgba(77,184,176,.4));animation:float 3s ease-in-out infinite;}
@keyframes float{0%,100%{transform:translateY(0);}50%{transform:translateY(-8px);}}
.welcome-hero h2{font-family:'Quicksand',sans-serif;font-size:2rem;font-weight:700;color:var(--text);margin:0;}
.welcome-hero p{font-size:.95rem;color:var(--text-muted);max-width:480px;line-height:1.6;margin:0;}
.stButton>button{background:#fff!important;color:#2d2926!important;border:1.5px solid #e8ddd3!important;border-radius:20px!important;font-family:'Nunito',sans-serif!important;font-weight:600!important;font-size:.84rem!important;padding:.5rem 1.1rem!important;width:100%!important;cursor:pointer!important;transition:border-color .15s,background .15s,color .15s,transform .1s!important;box-shadow:0 2px 16px rgba(0,0,0,.07)!important;}
[data-testid="stSidebar"] .stButton>button{background:linear-gradient(135deg,#f4845f 0%,#e86c47 100%)!important;color:#fff!important;border:none!important;border-radius:10px!important;}
[data-testid="stChatInput"] textarea{border-radius:var(--radius)!important;border:1.5px solid var(--border)!important;background:var(--surface)!important;font-family:'Nunito',sans-serif!important;font-size:.9rem!important;color:var(--text)!important;padding:.75rem 1.1rem!important;box-shadow:var(--shadow)!important;}
[data-testid="stChatInputSubmitButton"] button{background:var(--accent)!important;border-radius:50%!important;width:38px!important;height:38px!important;border:none!important;}
.status-bar{text-align:center;font-size:.72rem;color:var(--text-muted);padding:.5rem 0 .2rem;display:flex;align-items:center;justify-content:center;gap:.4rem;}
.status-dot{width:7px;height:7px;background:#4caf50;border-radius:50%;display:inline-block;animation:pulse 2s ease-in-out infinite;}
@keyframes pulse{0%,100%{opacity:1;}50%{opacity:.4;}}
/* ── Related-topic suggestion chips ── */
.related-label{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted);margin:0.55rem 0 0.3rem;}
div[data-rel-chip]>div>button{background:var(--teal-soft)!important;color:#2a7a75!important;border:1.5px solid #b2e4e1!important;border-radius:20px!important;font-size:.78rem!important;font-weight:600!important;padding:.28rem .85rem!important;width:auto!important;box-shadow:none!important;}
div[data-rel-chip]>div>button:hover{background:#b2e4e1!important;border-color:var(--teal)!important;color:#1d6662!important;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _load_doc_performance() -> list[dict]:
    """Load latest doc_performance.json if it exists."""
    path = cfg.reports_dir / "doc_performance.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ─────────────────────────────────────────────
# LDA ENGINE  (loaded once, cached for the session)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="🔬 Loading topic model…")
def load_lda_engine():
    """
    Fit an LDA topic model on the same parquet used for the vector DB.
    Returns (lda_model, vectorizer, topic_labels) or None if parquet missing.

    Topic labels are the top-3 keywords for each topic, used as human-readable
    fallback suggestions when the doc-title approach yields duplicates.
    """
    import nltk
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import CountVectorizer as CV
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    parquet_path = cfg.parquet_file
    if not parquet_path.exists():
        return None

    try:
        nltk.download("stopwords", quiet=True)
        sw = stopwords.words("english")
        sw.extend([
            "child", "kids", "health", "parent", "help", "children", "may",
            "use", "get", "doctor", "doctors", "people", "baby", "also",
            "make", "getting", "like", "type", "teen", "teens", "one",
            "know", "need", "time", "way", "ask",
        ])

        df = pd.read_parquet(parquet_path)
        texts = df["text"].astype(str).tolist()

        vectorizer = CV(max_df=0.95, min_df=2, stop_words=sw, max_features=3000)
        dtm = vectorizer.fit_transform(texts)

        lda = LDA(n_components=10, random_state=42, max_iter=15)
        lda.fit(dtm)

        # Pre-compute top-5 keywords per topic for quick lookup
        feature_names = vectorizer.get_feature_names_out()
        topic_keywords = []
        for topic in lda.components_:
            top_idx  = topic.argsort()[:-6:-1]   # top 5, descending
            keywords = [feature_names[i] for i in top_idx]
            topic_keywords.append(keywords)

        # Store doc-level dominant topics so we can find neighbours fast
        doc_topics   = lda.transform(dtm).argmax(axis=1)   # shape (n_docs,)
        df["_topic"] = doc_topics

        return {
            "lda":            lda,
            "vectorizer":     vectorizer,
            "topic_keywords": topic_keywords,   # list[list[str]], len=10
            "df":             df,               # parquet df with _topic column
        }
    except Exception as e:
        print(f"[LDA] Failed to build topic model: {e}")
        return None


def get_related_suggestions(docs: list, current_question: str, n: int = 3) -> list[dict]:
    """
    Return up to `n` related-topic suggestions after an AI response.

    Strategy (two-layer):
    1. PRIMARY — use the titles of retrieved docs, deduplicated and cleaned.
       Each unique title becomes a suggestion whose `query` asks about that article.
    2. FALLBACK — if fewer than `n` titles are available (e.g. all docs share one
       title), use the LDA topic most similar to the current question to surface
       keyword-based follow-up prompts.

    Returns:
        List of dicts: [{"label": str, "query": str}, ...]
    """
    suggestions: list[dict] = []
    seen_titles: set[str] = set()

    # ── Layer 1: doc titles ───────────────────────────────────────────────
    # Skip the title of the doc that was MOST relevant (already answered).
    # Use the 2nd, 3rd… titles as "related" suggestions.
    SKIP_WORDS = {"unknown", "pediatric article", "pediatric advice"}
    question_lower = current_question.lower()

    for doc in docs:
        title = doc.metadata.get("title", "").strip()
        url   = doc.metadata.get("source", "")

        if not title or title.lower() in SKIP_WORDS:
            continue
        if title in seen_titles:
            continue
        # Don't suggest a title whose keywords already appear in the question
        title_words = set(title.lower().split())
        if len(title_words & set(question_lower.split())) >= 2:
            continue

        seen_titles.add(title)
        suggestions.append({
            "label": title[:55] + ("…" if len(title) > 55 else ""),
            "query": f"Tell me more about: {title}",
            "url":   url,
        })

        if len(suggestions) >= n:
            break

    # ── Layer 2: LDA fallback ─────────────────────────────────────────────
    if len(suggestions) < n:
        engine = load_lda_engine()
        if engine:
            try:
                vec   = engine["vectorizer"]
                lda   = engine["lda"]
                kws   = engine["topic_keywords"]

                q_vec      = vec.transform([current_question])
                topic_dist = lda.transform(q_vec)[0]          # (10,)
                # Try topics in probability order, skipping dominant one
                sorted_topics = topic_dist.argsort()[::-1]

                for topic_idx in sorted_topics[1:]:            # skip the top match
                    keywords = kws[topic_idx]
                    label    = " · ".join(keywords[:3]).title()
                    query    = f"What do you know about {keywords[0]} and {keywords[1]} in children?"

                    if label not in seen_titles:
                        seen_titles.add(label)
                        suggestions.append({"label": label, "query": query, "url": ""})

                    if len(suggestions) >= n:
                        break
            except Exception as e:
                print(f"[LDA suggest] {e}")

    return suggestions[:n]


def render_message(role: str, content: str, reasoning: str = "", docs=None, timestamp: str = ""):
    """Render a chat bubble, and for AI messages show reasoning + sources."""
    avatar_emoji = "👤" if role == "user" else "👶"
    bubble_class = "user" if role == "user" else "ai"
    row_class    = "user" if role == "user" else "assistant"
    avatar_class = "user" if role == "user" else "ai"

    time_html = f'<div class="msg-time">{timestamp}</div>' if timestamp else ""
    st.markdown(
        f'<div class="msg-row {row_class}">'
        f'  <div class="avatar {avatar_class}">{avatar_emoji}</div>'
        f'  <div>'
        f'    <div class="bubble {bubble_class}">{content}</div>'
        f'    {time_html}'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Chain-of-Thought expander ─────────────────────────────────────────
    if role == "assistant" and reasoning and reasoning.strip():
        # Don't show the fallback "unavailable" notice as an expander
        is_fallback = reasoning.startswith("[Reasoning unavailable")
        label = "🧠 Show Reasoning Process" if not is_fallback else "⚠️ Reasoning unavailable"
        with st.expander(label, expanded=False):
            if is_fallback:
                st.warning(reasoning)
            else:
                # reasoning may contain **bold** labels (from dict normalisation)
                # or bullet lines (from list normalisation) — use st.markdown.
                # We also wrap it in a styled container for the monospace look.
                st.markdown(
                    "<style>"
                    ".reasoning-md p, .reasoning-md li { font-size: 0.83rem; line-height: 1.7; }"
                    ".reasoning-md { background:#f8f4ff; border:1.5px solid #9b7fca; "
                    "border-radius:10px; padding:.85rem 1.1rem; color:#4a3a6b; }"
                    "</style>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="reasoning-md">{reasoning}</div>',
                    unsafe_allow_html=True,
                )

    # ── Sources expander ──────────────────────────────────────────────────
    if role == "assistant" and docs and not bot._is_a_negative_answer(content, docs):
        seen, sources = set(), []
        for doc in docs:
            url   = doc.metadata.get("source", "")
            title = doc.metadata.get("title", "Pediatric Article")
            if url and url not in seen:
                sources.append((title, url))
                seen.add(url)
        if sources:
            with st.expander(f"📄 {len(sources)} source{'s' if len(sources)>1 else ''} consulted"):
                for title, url in sources:
                    st.markdown(f"- [{title}]({url})")
            st.caption(f"Retrieved {len(docs)} chunks · {len(sources)} documents")


# ─────────────────────────────────────────────
# BACKEND (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_backend():
    return RAGChat()

bot = load_backend()


# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "messages"           not in st.session_state: st.session_state.messages           = []
if "conversations"      not in st.session_state: st.session_state.conversations      = []
if "pending_suggestion" not in st.session_state: st.session_state.pending_suggestion = None
if "active_tab"         not in st.session_state: st.session_state.active_tab         = "chat"


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="logo-icon">👶</div>
        <h1>BabyChatty</h1>
        <div class="tagline">Pediatric AI Assistant<br>powered by Llama 3.1 · RAG</div>
    </div>
    """, unsafe_allow_html=True)

    # Tab selector
    tab_col1, tab_col2 = st.columns(2)
    with tab_col1:
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.active_tab = "chat"
    with tab_col2:
        if st.button("📊 Analytics", use_container_width=True):
            st.session_state.active_tab = "analytics"

    st.markdown("---")

    if st.session_state.active_tab == "chat":
        # ── New Chat ──────────────────────────────────────────────────────
        if st.button("✏️  New conversation"):
            if st.session_state.messages:
                first_msg = next(
                    (m["content"][:40] for m in st.session_state.messages if m["role"] == "user"),
                    "Conversation"
                )
                st.session_state.conversations.insert(0, {
                    "label": first_msg,
                    "messages": list(st.session_state.messages),
                })
            st.session_state.messages = []
            st.rerun()

        if st.session_state.conversations:
            st.markdown('<div style="font-size:.72rem;font-weight:700;text-transform:uppercase;color:#8c7b72;padding:.4rem 0">Recent</div>', unsafe_allow_html=True)
            for i, conv in enumerate(st.session_state.conversations[:8]):
                label = conv["label"] + ("…" if len(conv["label"]) == 40 else "")
                if st.button(f"💬 {label}", key=f"conv_{i}"):
                    st.session_state.messages = list(conv["messages"])
                    st.rerun()

    else:
        # ── Analytics tab ─────────────────────────────────────────────────
        st.markdown("### 📊 Document Performance")
        doc_perf = _load_doc_performance()

        if not doc_perf:
            st.info("No evaluation data yet.\nRun `main.py` to generate results.")
        else:
            valid = [d for d in doc_perf if d.get("avg_faithfulness") is not None]
            if not valid:
                st.warning("Evaluation ran but no Ragas scores were recorded.")
            else:
                top_doc    = valid[0]   # already sorted desc by faithfulness
                bottom_doc = valid[-1]

                st.markdown("**🏆 Top Performing Document**")
                st.markdown(
                    f'<div class="doc-card top">'
                    f'<span class="doc-score" style="color:#4caf50">'
                    f'{top_doc["avg_faithfulness"]:.3f}</span>'
                    f'<b>{top_doc["title"][:50]}</b><br>'
                    f'<a href="{top_doc["source"]}" target="_blank" style="font-size:.72rem">'
                    f'{top_doc["source"][:60]}…</a><br>'
                    f'<span style="font-size:.72rem;color:#666">Appearances: {top_doc["appearances"]} | '
                    f'Relevancy: {top_doc["avg_answer_relevancy"]:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown("**⚠️ Lowest Performing Document**")
                st.markdown(
                    f'<div class="doc-card bottom">'
                    f'<span class="doc-score" style="color:#f44336">'
                    f'{bottom_doc["avg_faithfulness"]:.3f}</span>'
                    f'<b>{bottom_doc["title"][:50]}</b><br>'
                    f'<a href="{bottom_doc["source"]}" target="_blank" style="font-size:.72rem">'
                    f'{bottom_doc["source"][:60]}…</a><br>'
                    f'<span style="font-size:.72rem;color:#666">Appearances: {bottom_doc["appearances"]} | '
                    f'Relevancy: {bottom_doc["avg_answer_relevancy"]:.3f}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(f"*{len(valid)} unique sources tracked.*")

    # ── Info chips ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0 .6rem">
        <div style="font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:#8c7b72;margin-bottom:.5rem">System</div>
        <span style="display:inline-flex;align-items:center;gap:.35rem;background:#d9f2f0;color:#4db8b0;border-radius:20px;font-size:.72rem;font-weight:700;padding:.3rem .75rem;margin:.2rem .3rem">🧠 {cfg.model_name}</span>
        <span style="display:inline-flex;align-items:center;gap:.35rem;background:#d9f2f0;color:#4db8b0;border-radius:20px;font-size:.72rem;font-weight:700;padding:.3rem .75rem;margin:.2rem .3rem">🔍 RAG · ChromaDB</span>
    </div>
    <div style="padding:.8rem .9rem 0;font-size:.72rem;color:#8c7b72;line-height:1.5;">
        ⚠️ General pediatric info only. Always consult your pediatrician.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="status-bar" style="margin-top:1rem">
        <span class="status-dot"></span> Model connected
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────
SUGGESTIONS = [
    ("🤒", "My baby has a fever of 38.5°C. What should I do?"),
    ("🍼", "When can I start introducing solid foods?"),
    ("😴", "How many hours should a 6-month-old sleep?"),
    ("💉", "Which vaccines does my baby need at 12 months?"),
    ("😢", "My toddler won't stop crying. Could it be colic?"),
    ("🌡️", "What are signs of dehydration in infants?"),
]

if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-hero">
        <div class="hero-icon">👶</div>
        <h2>How can I help today?</h2>
        <p>I'm BabyChatty, your pediatric assistant. Ask me anything about your child's health, development, or nutrition.</p>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, (icon, text) in enumerate(SUGGESTIONS):
        with cols[idx % 2]:
            if st.button(f"{icon}  {text}", key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_suggestion = text
                st.rerun()
else:
    for msg in st.session_state.messages:
        render_message(
            role      = msg["role"],
            content   = msg["content"],
            reasoning = msg.get("reasoning", ""),
            docs      = msg.get("docs"),
        )


# ─────────────────────────────────────────────
# CHAT INPUT & LOGIC
# ─────────────────────────────────────────────
prompt = st.chat_input("Ask me about your child's health…")

if st.session_state.pending_suggestion:
    prompt = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(role="user", content=prompt)

    with st.spinner("🔍 Searching medical database…"):
        try:
            lang = bot._detect_language(prompt)
            reasoning, answer, docs = bot._get_ai_response(prompt, lang)
            if not bot._is_a_negative_answer(answer, docs):
                answer += "\n" + cfg.disclamer_prompt[lang]

        except ollama.ResponseError as e:
            reasoning = ""
            answer = f"⚠️ Ollama API Error: {e.error}"
            docs = []
        except Exception as e:
            reasoning = ""
            answer = f"⚠️ Connection error: {e}"
            docs = []

    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "reasoning": reasoning,
        "docs":      docs,
    })
    st.rerun()