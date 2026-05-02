import sys, os, json, time
from pathlib import Path

# Fix path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import ollama
import pandas as pd
import warnings
from utils import RAGChat, GenConfig as cfg

# Silence transformers warnings in the terminal
warnings.filterwarnings("ignore", module="transformers")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BabyChatty AI",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS for Styling ──
st.markdown("""
<style>
    .welcome-hero { text-align: center; padding: 2rem 0; background-color: #f8f9fa; border-radius: 15px; margin-bottom: 2rem; }
    .hero-icon { font-size: 4rem; margin-bottom: 1rem; }
    .related-header { margin-top: 20px; font-weight: bold; color: #555; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SYSTEM & STATE INITIALIZATION
# ─────────────────────────────────────────────
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
    prompt = f"""
        Based on this medical context, generate EXACTLY 4 short, follow-up questions a parent might ask.
        CRITICAL RULES:
        1. Return ONLY the 4 questions, one per line.
        2. DO NOT include any introductory text.
        3. DO NOT include numbers or bullet points.
        4. Language MUST be: {lang}.
        
        Context: {context}
    """
    try:
        response = bot.client.generate(
            model=cfg.model_name, 
            prompt=prompt, 
            options={"num_predict": 150, "temperature": cfg.temperature}
        )
        
        raw_text = response['response']
        questions = []
        for line in raw_text.split('\n'):
            clean_line = line.strip(" 1234567890.-*•")
            if len(clean_line) > 10 and '?' in clean_line:
                questions.append(clean_line)
        
        return questions[:4]
    except:
        return ["Tell me more about symptoms.", "How can I prevent this?", "When should I call a doctor?", "What are the treatment options?"]

# ─────────────────────────────────────────────
# MESSAGE RENDERING
# ─────────────────────────────────────────────
def render_message(msg_idx, role, content, reasoning="", docs=None, related_qs=None, stats=None):
    with st.chat_message(role):
        if role == "assistant" and reasoning:
            with st.expander("🧠 Internal Reasoning"):
                st.info(reasoning)
        
        st.markdown(content)
        
        if role == "assistant" and docs:
            # Slicer: Source Analysis
            show_analysis = st.toggle("Show Source Analysis", key=f"analysis_toggle_{msg_idx}")
            if show_analysis:
                st.markdown("---")
                with st.spinner("Summarizing context..."):
                    lang = bot._detect_language(content)
                    st.markdown(f"**Context Summary:**\n{bot.summarize_docs(docs, lang)}")
                
                st.markdown("**Retrieved Documents & Relevance:**")
                for i, doc in enumerate(docs):
                    score = doc.metadata.get("relevance_score", 0.0)
                    title = doc.metadata.get("title", "Document")
                    url = doc.metadata.get("source", "#")
                    st.markdown(f"- **[{score:.3f}]** [{title}]({url})")

            # Metrics Line (NEW: Gray line with counts and time)
            if stats:
                st.caption(f"⚡ {stats['chunks']} chunks | 📚 {stats['docs']} docs | ⏱️ {stats['time']:.2f}s")

            # Related Questions Bubbles
            if related_qs:
                st.markdown('<p class="related-header">Related Questions:</p>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, q_text in enumerate(related_qs):
                    with cols[i % 2]:
                        if st.button(f"🔍 {q_text}", key=f"rel_{msg_idx}_{i}", use_container_width=True):
                            st.session_state.pending_suggestion = q_text
                            st.rerun()

# ─────────────────────────────────────────────
# MAIN UI FLOW
# ─────────────────────────────────────────────
SUGGESTIONS = [
    ("🤒", "My baby has a fever of 38.5°C. What should I do?"),
    ("💉", "Which vaccines does my baby need at 12 months?"),
    ("🦠", "What are the early signs of Hand, Foot and Mouth disease?"),
    ("🌡️", "How do I correctly take a rectal temperature?"),
]

if not st.session_state.messages:
    st.markdown('<div class="welcome-hero"><div class="hero-icon">👶</div><h2>How can I help today?</h2><p>Ask BabyChatty about pediatric infections and vaccines.</p></div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, (icon, text) in enumerate(SUGGESTIONS):
        with cols[idx % 2]:
            if st.button(f"{icon}  {text}", key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_suggestion = text
                st.rerun()
else:
    for idx, msg in enumerate(st.session_state.messages):
        render_message(idx, msg["role"], msg["content"], msg.get("reasoning"), msg.get("docs"), msg.get("related_qs"), msg.get("stats"))

# ── Chat Input ──
prompt = st.chat_input("Ask me about your child's health…")

if st.session_state.pending_suggestion:
    prompt = st.session_state.pending_suggestion
    st.session_state.pending_suggestion = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

# ── Processing logic ──
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_prompt = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching medical database…"):
            start_time = time.time()
            try:
                lang = bot._detect_language(last_prompt)
                reasoning, answer, docs = bot._get_ai_response(last_prompt, lang)
                end_time = time.time()

                # Performance Stats
                stats = {
                    "chunks": len(docs) if docs else 0,
                    "docs": len(set(d.metadata.get("source") for d in docs)) if docs else 0,
                    "time": end_time - start_time
                }
                
                # Generate Related Questions
                context_str = " ".join([d.page_content for d in docs]) if docs else ""
                related_qs = get_related_questions(context_str, lang) if docs else []
                
                if not bot._is_a_negative_answer(answer, docs):
                    answer += "\n\n" + cfg.disclamer_prompt.get(lang, cfg.disclamer_prompt["English"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "reasoning": reasoning,
                    "docs": docs,
                    "related_qs": related_qs,
                    "stats": stats
                })
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ System error: {e}")

# ── Sidebar ──
st.sidebar.markdown("### System Status")
st.sidebar.success("Database Connected")
st.sidebar.info(f"LLM: {cfg.model_name}")