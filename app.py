import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import ollama # Needed to catch specific API errors
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
# GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Quicksand:wght@500;600;700&display=swap');

/* ── Root tokens ── */
:root {
    --bg:           #f7f4f0;
    --surface:      #ffffff;
    --surface-2:    #fdf6ee;
    --border:       #e8ddd3;
    --accent:       #f4845f;
    --accent-soft:  #fde8df;
    --teal:         #4db8b0;
    --teal-soft:    #d9f2f0;
    --purple:       #9b7fca;
    --purple-soft:  #ede6f8;
    --text:         #2d2926;
    --text-muted:   #8c7b72;
    --user-bubble:  #4db8b0;
    --ai-bubble:    #ffffff;
    --radius:       18px;
    --radius-sm:    10px;
    --shadow:       0 2px 16px rgba(0,0,0,.07);
    --shadow-md:    0 4px 24px rgba(0,0,0,.11);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'Nunito', sans-serif;
    color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1.5px solid var(--border) !important;
    padding-top: 0 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Sidebar logo area ── */
.sidebar-logo {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 1.5rem 1rem 1rem;
    border-bottom: 1.5px solid var(--border);
    margin-bottom: 1.2rem;
}
.sidebar-logo .logo-icon {
    font-size: 3.2rem;
    line-height: 1;
    margin-bottom: .4rem;
    filter: drop-shadow(0 2px 8px rgba(244,132,95,.35));
}
.sidebar-logo h1 {
    font-family: 'Quicksand', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
    letter-spacing: -.3px;
}
.sidebar-logo .tagline {
    font-size: .75rem;
    color: var(--text-muted);
    margin-top: .2rem;
    text-align: center;
    line-height: 1.4;
}

/* ── Sidebar section labels ── */
.sidebar-section {
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--text-muted);
    padding: 0 1.2rem .4rem;
    margin-top: .8rem;
}

/* ── Sidebar CTA button ── */
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] button[kind="primary"],
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #f4845f 0%, #e86c47 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: .88rem !important;
    padding: .55rem 1.2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity .15s, transform .1s !important;
    box-shadow: 0 2px 8px rgba(244,132,95,.35) !important;
}
[data-testid="stSidebar"] .stButton > button * { color: #ffffff !important; }
[data-testid="stSidebar"] .stButton > button:hover { opacity: .9 !important; transform: translateY(-1px) !important; }

/* ── Suggestion chip buttons (main content) ── */
.stButton > button,
div[data-testid="stButton"] > button,
button[kind="secondary"],
button[kind="primary"] {
    background: #ffffff !important;
    color: #2d2926 !important;
    border: 1.5px solid #e8ddd3 !important;
    border-radius: 20px !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    font-size: .84rem !important;
    padding: .5rem 1.1rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: border-color .15s, background .15s, color .15s, transform .1s !important;
    box-shadow: 0 2px 16px rgba(0,0,0,.07) !important;
}
.stButton > button *, 
div[data-testid="stButton"] > button *,
button[kind="secondary"] *,
button[kind="primary"] * {
    color: #2d2926 !important;
}
.stButton > button:hover,
div[data-testid="stButton"] > button:hover,
button[kind="secondary"]:hover,
button[kind="primary"]:hover {
    border-color: #4db8b0 !important;
    background: #d9f2f0 !important;
    color: #4db8b0 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(77,184,176,.2) !important;
}
.stButton > button:hover *,
div[data-testid="stButton"] > button:hover *,
button[kind="secondary"]:hover *,
button[kind="primary"]:hover * {
    color: #4db8b0 !important;
}

/* Sidebar overrides come AFTER to win specificity */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #f4845f 0%, #e86c47 100%) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(244,132,95,.35) !important;
}
[data-testid="stSidebar"] .stButton > button *,
[data-testid="stSidebar"] div[data-testid="stButton"] > button * { color: #ffffff !important; }
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #f4845f 0%, #e86c47 100%) !important;
    opacity: .88 !important;
    border: none !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton > button:hover *,
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover * { color: #ffffff !important; }

/* ── Chat input cursor ── */
[data-testid="stChatInput"] textarea {
    caret-color: var(--teal) !important;
}

/* ── Conversation history items ── */
.conv-item {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .55rem 1.2rem;
    border-radius: var(--radius-sm);
    cursor: pointer;
    font-size: .85rem;
    color: var(--text);
    margin: .15rem .6rem;
    transition: background .15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.conv-item:hover { background: var(--surface-2); }
.conv-item.active { background: var(--accent-soft); color: var(--accent); font-weight: 700; }
.conv-item .conv-icon { font-size: 1rem; flex-shrink: 0; }

/* ── Sidebar info chips ── */
.info-chip {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    background: var(--teal-soft);
    color: var(--teal);
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 700;
    padding: .3rem .75rem;
    margin: .2rem .3rem;
}

/* ── Main chat area ── */
.main-wrap {
    max-width: 780px;
    margin: 0 auto;
    padding: 0 1rem 120px;
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ── Welcome hero (shows when no messages) ── */
.welcome-hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 4rem 1rem 2rem;
    gap: 1rem;
}
.welcome-hero .hero-icon {
    font-size: 4rem;
    filter: drop-shadow(0 4px 16px rgba(77,184,176,.4));
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-8px); }
}
.welcome-hero h2 {
    font-family: 'Quicksand', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.welcome-hero p {
    font-size: .95rem;
    color: var(--text-muted);
    max-width: 480px;
    line-height: 1.6;
    margin: 0;
}

/* ── Suggestion chips ── */
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: .6rem;
    justify-content: center;
    padding: .5rem 0 1.5rem;
}
.chip {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: 20px;
    padding: .45rem 1rem;
    font-size: .82rem;
    font-weight: 600;
    color: var(--text);
    cursor: pointer;
    transition: border-color .15s, background .15s, transform .1s;
    display: inline-flex;
    align-items: center;
    gap: .35rem;
}
.chip:hover {
    border-color: var(--teal);
    background: var(--teal-soft);
    color: var(--teal);
    transform: translateY(-1px);
}

/* ── Message row ── */
.msg-row {
    display: flex;
    align-items: flex-end;
    gap: .75rem;
    padding: .6rem 0;
    animation: fadeUp .25s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.msg-row.user  { flex-direction: row-reverse; }
.msg-row.assistant { flex-direction: row; }

/* ── Avatars ── */
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,.1);
}
.avatar.user { background: linear-gradient(135deg, var(--teal) 0%, #3aa39b 100%); }
.avatar.ai   { background: linear-gradient(135deg, var(--accent) 0%, #e86c47 100%); }

/* ── Bubbles ── */
.bubble {
    max-width: 72%;
    padding: .85rem 1.15rem;
    border-radius: var(--radius);
    font-size: .92rem;
    line-height: 1.65;
    box-shadow: var(--shadow);
    position: relative;
}
.bubble.user {
    background: var(--user-bubble);
    color: #fff;
    border-bottom-right-radius: 4px;
}
.bubble.ai {
    background: var(--ai-bubble);
    color: var(--text);
    border: 1.5px solid var(--border);
    border-bottom-left-radius: 4px;
}

/* ── Timestamp ── */
.msg-time {
    font-size: .68rem;
    color: var(--text-muted);
    padding: 0 .3rem;
    align-self: flex-end;
    white-space: nowrap;
}

/* ── Disclaimer badge ── */
.disclaimer {
    display: inline-flex;
    align-items: center;
    gap: .3rem;
    background: var(--purple-soft);
    color: var(--purple);
    border-radius: 8px;
    font-size: .72rem;
    font-weight: 600;
    padding: .3rem .65rem;
    margin-top: .55rem;
}

/* ── Sources expander ── */
[data-testid="stExpander"] {
    background: var(--surface-2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-top: .4rem !important;
    font-size: .82rem !important;
}
[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: var(--teal) !important;
    font-size: .82rem !important;
}

/* ── Typing indicator ── */
.typing {
    display: flex;
    align-items: center;
    gap: .7rem;
    padding: .6rem 0;
}
.dots {
    display: flex;
    gap: .3rem;
    background: var(--ai-bubble);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    border-bottom-left-radius: 4px;
    padding: .75rem 1rem;
    box-shadow: var(--shadow);
}
.dots span {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce 1.1s ease-in-out infinite;
}
.dots span:nth-child(2) { animation-delay: .18s; }
.dots span:nth-child(3) { animation-delay: .36s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: .5; }
    40%            { transform: translateY(-7px); opacity: 1; }
}

/* ── Chat input override ── */
[data-testid="stChatInput"] {
    background: var(--bg) !important;
    border-top: 1.5px solid var(--border) !important;
    padding: .85rem 1.5rem 1rem !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: var(--radius) !important;
    border: 1.5px solid var(--border) !important;
    background: var(--surface) !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: .9rem !important;
    color: var(--text) !important;
    padding: .75rem 1.1rem !important;
    box-shadow: var(--shadow) !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--teal) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(77,184,176,.15), var(--shadow) !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background: var(--accent) !important;
    border-radius: 50% !important;
    width: 38px !important;
    height: 38px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(244,132,95,.4) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 10px; }

/* ── Caption / status bar ── */
.status-bar {
    text-align: center;
    font-size: .72rem;
    color: var(--text-muted);
    padding: .5rem 0 .2rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: .4rem;
}
.status-dot {
    width: 7px; height: 7px;
    background: #4caf50;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: .4; }
}
</style>
""", unsafe_allow_html=True)


def render_message(role: str, content: str, docs=None, timestamp: str = ""):
    """Renders a chat message with custom HTML/CSS for a native app feel."""
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

    # Sources expander (only for AI messages that successfully retrieved info)
    if role == "assistant" and not bot._is_a_negative_answer(content, docs):
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
# LOAD STATE & BACKEND CLASSES (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_backend():
    """
    Initializes the RAGChat class once per session.
    This automatically loads/creates the VectorDB using the config settings.
    """
    # Since VectorDB initialization happens inside RAGChat.__init__,
    # we just need to instantiate the bot class.
    return RAGChat()

# Initialize the RAG bot engine
bot = load_backend()

# ─────────────────────────────────────────────
# SESSION STATE MANAGEMENT
# ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # Format: [{"role": str, "content": str, "docs": list}]
if "conversations" not in st.session_state:
    st.session_state.conversations = []     # List of saved conversation snapshots
if "pending_suggestion" not in st.session_state:
    st.session_state.pending_suggestion = None


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

    # New chat button
    if st.button("✏️  New conversation"):
        if st.session_state.messages:
            # Save a snapshot of the current conversation
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

    # Conversation history
    if st.session_state.conversations:
        st.markdown('<div class="sidebar-section">Recent</div>', unsafe_allow_html=True)
        for i, conv in enumerate(st.session_state.conversations[:8]):
            label = conv["label"] + ("…" if len(conv["label"]) == 40 else "")
            if st.button(f"💬 {label}", key=f"conv_{i}"):
                st.session_state.messages = list(conv["messages"])
                st.rerun()

    # Info chips
    st.markdown("---")
    st.markdown(f"""
    <div style="padding:0 .6rem">
        <div style="font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-bottom:.5rem">System</div>
        <span class="info-chip">🧠 {cfg.model_name}</span>
        <span class="info-chip">🔍 RAG · ChromaDB</span>
        <span class="info-chip" style="background:var(--accent-soft);color:var(--accent)">👩‍⚕️ Pediatric DB</span>
    </div>
    <div style="padding:.8rem .9rem 0;font-size:.72rem;color:var(--text-muted);line-height:1.5;">
        ⚠️ This assistant provides <b>general pediatric information</b> only and does not replace professional medical advice. Always consult your pediatrician.
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
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

# Display Welcome hero if chat is empty
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-hero">
        <div class="hero-icon">👶</div>
        <h2>How can I help today?</h2>
        <p>I'm BabyChatty, your pediatric assistant. Ask me anything about your child's health, development, or nutrition.</p>
    </div>
    """, unsafe_allow_html=True)

    # Render suggestion buttons
    cols = st.columns(2)
    for idx, (icon, text) in enumerate(SUGGESTIONS):
        with cols[idx % 2]:
            label = f"{icon}  {text}"
            if st.button(label, key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_suggestion = text
                st.rerun()
else:
    # Render the conversation history
    for msg in st.session_state.messages:
        render_message(
            role    = msg["role"],
            content = msg["content"],
            docs    = msg.get("docs"),
        )


# ─────────────────────────────────────────────
# CHAT INPUT & LOGIC
# ─────────────────────────────────────────────
prompt = st.chat_input("Ask me about your child's health…")

# Check if a suggestion button was clicked
if st.session_state.pending_suggestion:
    prompt = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None

if prompt:
    # 1. Add and render the user's prompt immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    render_message(role="user", content=prompt)

    # 2. Query the backend while showing a spinner
    with st.spinner("🔍 Searching medical database…"):
        try:
            # We use the internal methods of the RAGChat instance
            lang = bot._detect_language(prompt)
            answer, docs = bot._get_ai_response(prompt, lang)
            if not bot._is_a_negative_answer(answer, docs):   # add the disclamer
                answer += "\n\n"+cfg.disclamer_prompt[lang]

        except ollama.ResponseError as e:
            # Specific handling for API keys or connection refusals (e.g. 401 Unauthorized)
            answer = f"⚠️ Ollama API Error: The server rejected the request. Details: {e.error}"
            docs = []
            
        except Exception as e:
            # General fallback
            answer = f"⚠️ Connection error: Please check your network or API keys. Details: {e}"
            docs = []

    # 3. Save and render the assistant's response
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "docs":    docs,
    })

    st.rerun()