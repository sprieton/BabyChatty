import streamlit as st
import ollama
from utils import RAGChat, GenConfig as cfg

# --- Page Configuration ---
st.set_page_config(
    page_title="BabyChatty - Pediatric Assistant",
    page_icon="👶",
    layout="wide"
)

# --- Custom Styles ---
st.markdown("""
<style>
    .welcome-hero {
        text-align: center;
        padding: 3rem 0;
        background-color: #f8f9fa;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .hero-icon { font-size: 4rem; margin-bottom: 1rem; }
    .related-header { margin-top: 20px; font-weight: bold; color: #555; }
</style>
""", unsafe_allow_html=True)

# --- System & State Initialization ---
if 'baby_chatty' not in st.session_state:
    with st.spinner("Initializing Pediatric RAG System..."):
        st.session_state.baby_chatty = RAGChat()

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pending_suggestion' not in st.session_state:
    st.session_state.pending_suggestion = None

bot = st.session_state.baby_chatty

# --- Helper: Generate Related Questions ---
def get_related_questions(context, lang):
    """Uses the LLM to generate 4 related questions based on the context."""
    prompt = f"Based on this medical context, generate 4 short, follow-up questions a parent might ask. Language: {lang}. Context: {context[:2000]}"
    try:
        response = bot.client.generate(model=cfg.model_name, prompt=prompt, options={"num_predict": 100})
        # Simple parsing: split by lines/numbers and take first 4
        questions = [q.strip(" 1234.-") for q in response['response'].split('\n') if len(q.strip()) > 10][:4]
        return questions
    except:
        return ["Tell me more about symptoms.", "How can I prevent this?", "When should I call a doctor?", "What are the treatment options?"]

# --- Function to Render Chat Messages ---
def render_message(msg_idx, role, content, reasoning="", docs=None, related_qs=None):
    with st.chat_message(role):
        if role == "assistant" and reasoning:
            with st.expander("🧠 Internal Reasoning"):
                st.info(reasoning)
        
        st.markdown(content)
        
        if role == "assistant" and docs:
            # --- THE SLICER: Show Analysis ---
            show_analysis = st.toggle("Show Source Analysis", key=f"analysis_toggle_{msg_idx}")
            
            if show_analysis:
                st.markdown("---")
                # 1. Show Context Summary
                with st.spinner("Summarizing context..."):
                    lang = bot._detect_language(content)
                    summary = bot.summarize_docs(docs, lang)
                    st.markdown(f"**Context Summary:**\n{summary}")
                
                # 2. Show Documents with Scores
                st.markdown("**Retrieved Documents & Relevance:**")
                for i, doc in enumerate(docs):
                    score = doc.metadata.get("relevance_score", 0.0)
                    title = doc.metadata.get("title", "Document")
                    st.write(f"- **[{score:.3f}]** {title}")

            # --- RELATED QUESTIONS BUBBLES ---
            if related_qs:
                st.markdown('<p class="related-header">Related Questions:</p>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, q_text in enumerate(related_qs):
                    with cols[i % 2]:
                        if st.button(f"🔍 {q_text}", key=f"rel_{msg_idx}_{i}", use_container_width=True):
                            st.session_state.pending_suggestion = q_text
                            st.rerun()

# --- Suggestions Data (Initial State) ---
SUGGESTIONS = [
    ("🤒", "My baby has a fever of 38.5°C. What should I do?"),
    ("💉", "Which vaccines does my baby need at 12 months?"),
    ("🦠", "What are the early signs of Hand, Foot and Mouth disease?"),
    ("🌡️", "How do I correctly take a rectal temperature?"),
]

# --- Main UI Flow ---
if not st.session_state.messages:
    st.markdown('<div class="welcome-hero"><div class="hero-icon">👶</div><h2>How can I help today?</h2><p>Ask me about pediatric infections and vaccines.</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (icon, text) in enumerate(SUGGESTIONS):
        with cols[idx % 2]:
            if st.button(f"{icon}  {text}", key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_suggestion = text
                st.rerun()
else:
    for idx, msg in enumerate(st.session_state.messages):
        render_message(idx, msg["role"], msg["content"], msg.get("reasoning", ""), msg.get("docs"), msg.get("related_qs"))

# --- Chat Input Logic ---
prompt = st.chat_input("Ask BabyChatty...")

if st.session_state.pending_suggestion:
    prompt = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# Processing the latest message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Consulting the medical database..."):
            try:
                lang = bot._detect_language(last_prompt)
                reasoning, answer, docs = bot._get_ai_response(last_prompt, lang)
                
                # Generate Related Questions based on retrieved context
                context_str = " ".join([d.page_content for d in docs]) if docs else ""
                related_qs = get_related_questions(context_str, lang) if docs else []
                
                if not bot._is_a_negative_answer(answer, docs):
                    answer += "\n\n" + cfg.disclamer_prompt.get(lang, cfg.disclamer_prompt["English"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "reasoning": reasoning,
                    "docs": docs,
                    "related_qs": related_qs
                })
                st.rerun()
            except Exception as e:
                st.error(f"System Error: {e}")