import os, sys, time, re, json, statistics
import ollama
import logging
from typing import Sequence, Optional
import torch
from transformers import pipeline
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # checks if the answer is supported by the retrieved context
    answer_relevancy,   # checks if the answer is relevant to the question
    ContextUtilization  # checks if the context is relevant to the question
    )
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.callbacks import Callbacks

from utils import GenConfig as cfg
from utils import VectorDBFactory

class RAGChat:
    """
    RAG chat class with Chain-of-Thought reasoning, document-level metrics,
    and multilingual performance tracking.
    """

    def __init__(self):
        # 1. Load the .env file to get the API key and start the Ollama client
        load_dotenv()
        self.client = ollama.Client(
            host=cfg.ollama_url,
            headers={"X-API-KEY": os.getenv("OLLAMA_API_KEY")},
        )

        # 2. Initialize the LLM and the embeddings model for evaluation (Ragas)
        self.eval_llm = ChatOpenAI(
            base_url=f"{cfg.ollama_url}/v1",
            api_key="fake-key",  # same API key in the headers of the client
            model=cfg.judge_name,
            default_headers={"X-API-KEY": os.getenv("OLLAMA_API_KEY")},
            temperature=0,      # for evaluation, we want deterministic responses
            timeout=120         # avoid timeouts
        )
        logging.getLogger("ragas").setLevel(logging.ERROR)

        # 3. get the vector database ready, if not provided, create it
        print(f"[RAGChat] Loading vector database from {cfg.chroma_dir}...")
        try:
            self.vectorDB = VectorDBFactory(cfg.chroma_dir).get_vectorDB()
        except Exception as e:
            print(f"[RagChat] Error loading database: {e}")
            sys.exit(1)
        print(f"[RAGChat] 📚 Vector database done")

        # 4. define the embeddings model and the reranker
        # important to use models capable of multilingual embeddings
        if cfg.emb_device == "cuda":
            torch.cuda.empty_cache()
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            model_kwargs={'device': cfg.emb_device},
            encode_kwargs={"batch_size": 1}
            )

        # 4.1 First pull of retrieval with threshold
        base_retriever = self.vectorDB.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "score_threshold": cfg.ret_threshold,
                "k": cfg.max_ret_num    # max num of chunks to retrieve
            }
        )

        # 4.2 Reranker the retrieved chunks to the best 5
        compressor = CustomCrossEncoderReranker(
            model=HuggingFaceCrossEncoder(
                model_name=cfg.re_rank_model,
                model_kwargs={'device': cfg.emb_device} # same device as embeddings
                ),
            top_n=cfg.retrieval_num)
        self.retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

        # to store the conversation for coherence in the conversation
        self.chat_history = {
            "questions": [],
            "answers": [],
            "reasoning_traces": [],
            "retrieved_docs": []
        }
        self.metrics_history = {
            "response_times": [],
            "coverage": [],
            "faithfulness": [],
            "relevance": [],
            "context_util": []
        }

        # ── Document-level metrics accumulator ──────────────────────────────
        # Maps source_url -> list of per-turn metric scores
        self.doc_metrics: dict[str, dict[str, list]] = {}
    

    def start(self, eval_mode=False):
        """
        Start the RAG chat loop in the terminal.

        ---
        Attributes:
            - eval_mode: if True, the chat will evaluate responses
        """

        print(f"[RAGChat] -- UC3M Pediatric Bot Connected ({cfg.model_name}) ---")
        while True:
            user_input = input("\nParent: ")
            if user_input.lower() in ['exit', 'quit']: 
                break

            try:
                negative_phrase = False
                eval_results = None
                # 1. Detect language
                start_time = time.time()
                lang = self._detect_language(user_input)
                print(f"[RAGChat] Language detected: {lang}")

                # 2. Get AI response
                reasoning, answer, docs = self._get_ai_response(user_input, lang)
                generation_time = time.time() - start_time
                
                print(f"Reasoning: {reasoning}")
                print(f"\nAI: {answer}")

                # 3. Check if the answer is negative
                negative_phrase = self._is_a_negative_answer(answer, docs)

                # 4. Evaluate turn with Ragas (only if the answer is not negative)
                if not negative_phrase:
                    answer += f"""
                    \n\n{cfg.disclamer_prompt[lang]} \nSummary:\n {self.summarize_docs(docs)}
                    """
                    if eval_mode:
                        eval_results = self._evaluate_turn(user_input, answer, docs)

                # 5. Print the docs used for the answer (for traceability)
                # ── Stamp docs coverage if we have docs ─────────────────────────────────────────────────
                if not negative_phrase and docs:
                    self._stamp_context_docs(docs)

                # 6. Show and store the metrics
                if eval_mode:
                    self._stamp_and_store_metrics(
                        generation_time, eval_results, negative_phrase)
                
                # Store the answer for future context in the conversation
                self.chat_history["questions"].append(user_input)
                self.chat_history["answers"].append(answer)
                self.chat_history["reasoning_traces"].append(reasoning)
                self.chat_history["retrieved_docs"].append(docs)

            except ollama.ResponseError as e:
                print(f"\n[RAGChat] Ollama response error: {e.error}")
            except Exception as e:
                print(f"\n[RAGChat] connection error: {e}")


    def eval_questions(self, questions: dict):
        """
        Evaluate a batch of questions internally using the full RAG pipeline.
        
        --- 
        Attributes:
            - questions: dictionary of questions to evaluate with their label.

        ---
        Returns:
            List of dicts, each containing: query, language, reasoning,
            final_answer, retrieved_docs (with metadata), and Ragas metrics.
        """
        detailed_results = []
        results_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "faithfulness": [],
            "relevancy": [],
            "context_util": [],
        }

        total_time = 0
        TP = FP = TN = FN = 0   # confusion matrix

        for item in tqdm(questions, desc="Evaluating questions"):
            record = {
                "question": None,
                "language": None,
                "label": None,
                "reasoning": None,
                "final_answer": None,
                "retrieved_docs": [],   # list of {source, title, relevance_score, content}
                "is_negative": None,
                "faithfulness": None,
                "answer_relevancy": None,
                "context_utilization": None,
                "elapsed_s": None,
            }
            try:
                q = item["question"]
                label = item["label"]
                record["question"] = q
                record["label"] = label
                start_time = time.time()

                # 1. language detection
                lang = self._detect_language(q)
                record["language"] = lang

                # 2. RAG pipeline
                reasoning, answer, docs = self._get_ai_response(q, lang, indep_quest=True)
                elapsed = time.time() - start_time
                record["elapsed_s"] = round(elapsed, 3)
                record["reasoning"] = reasoning
                record["final_answer"] = answer
                record["retrieved_docs"] = [
                    {
                        "source": d.metadata.get("source", "unknown"),
                        "title": d.metadata.get("title", "unknown"),
                        "relevance_score": float(d.metadata.get("relevance_score", 0.0)),
                        "content": d.page_content[:500],  # truncate for storage
                    }
                    for d in docs
                ]

                # 3. skip negatives
                is_negative = self._is_a_negative_answer(answer, docs)
                record["is_negative"] = is_negative
                pred = 0 if is_negative else 1  # 0 = negative, 1 = positive
                # update the confusion matrix
                if label == 1:
                    TP += 1 if pred == 1 else 0
                    FN += 1 if pred == 0 else 0
                else:
                    FP += 1 if pred == 1 else 0
                    TN += 1 if pred == 0 else 0
                
                print(f"\tQuestion: {q} — {'positive' if label==1 else 'negative'}")
                if is_negative:
                    print("[RAGChat] Negative answer, skipping Ragas evaluation")
                    detailed_results.append(record)
                    continue

                # 4. evaluation
                eval_results = self._evaluate_turn(q, answer, docs, eval_mode=True)
                total_time += elapsed

                if eval_results is not None:
                    df = eval_results.to_pandas()
                    faith_score  = float(df["faithfulness"].iloc[0])
                    relev_score  = float(df["answer_relevancy"].iloc[0])
                    ctx_score    = float(df["context_utilization"].iloc[0])
 
                    record["faithfulness"]        = faith_score
                    record["answer_relevancy"]    = relev_score
                    record["context_utilization"] = ctx_score
 
                    # ── Map scores back to each source document ──────────────
                    self._update_doc_metrics(docs, faith_score, relev_score, ctx_score)

                detailed_results.append(record)

            except Exception as e:
                print(f"[eval_questions] Error processing question: {q}\n{e}")

        # ── Stamp summary ─────────────────────────────────────────────────────
        answered = [r for r in detailed_results if not r.get("is_negative")]
        results_data = {
            "faithfulness":  [r["faithfulness"]        for r in answered if r["faithfulness"] is not None],
            "relevancy":     [r["answer_relevancy"]     for r in answered if r["answer_relevancy"] is not None],
            "context_util":  [r["context_utilization"]  for r in answered if r["context_utilization"] is not None],
        }
        self._stamp_eval_metrics(results_data, total_time, TP, FP, TN, FN)
 
        return detailed_results


    # ──────────────────────────────────────────────────────────────────────────
    # ── Core RAG pipeline ─────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    def _get_ai_response(
            self, user_input, lang: str = "English", 
            indep_quest: bool = False) -> tuple[str, str, list]:
        """
        Implement the loop of RAG.
        1. Translation to english the query
        2. Retrieval        
        3. Build a CoT prompt
        4. Call model
        5. Parse structured output

        ---
        Attributes:
            - user_input: the question from the user
            - lang: the language of the user input
            - indep_quest: if True, the question is treated as independent
        ---
        Returns:
            (reasoning, answer, docs)
        """

        # 1. Translation to english the query
        if lang != "English":
            user_input = GoogleTranslator(source='auto', target='en').translate(user_input)

        # 2. Retrieval
        docs = self.retriever.invoke(user_input)

        # 2.1 Add source info to the context for better traceability:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown")
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {title} | {source}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # 2.2 Add the context of previous conversation turns
        if not indep_quest:
            prev_conversation = ""
            for q, a in zip(self.chat_history['questions'], self.chat_history['answers']):
                    prev_conversation += f"Parent: {q}\nAI: {a}\n\n"
            context = f"Previous conversation: {prev_conversation}\n\n" + context

        # 3. Build CoT prompt
        final_prompt = cfg.prompt_template.format(
            lang=lang,
            context=context,
            question=user_input
        )

        # 4. Call model
        response = self.client.chat(
            model=cfg.model_name,
            messages=[{"role": "user", "content": final_prompt}],
            options={"temperature": cfg.temperature}
        )
        raw = response["message"]["content"]
        # 5. Parse structured output
        parsed = _parse_cot_response(raw)
        return parsed["reasoning"], parsed["answer"], docs
    
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the user input and return the language name."""
        try:
            lang_code = detect(text)
            lang_map = {
                "es": "Spanish",
                "en": "English",
                "fr": "French",
                "de": "German",
                "it": "Italian",
                "pt": "Portuguese",
                "ca": "Catalan",
            }
            return lang_map.get(lang_code, "English")
        except LangDetectException:
            # fallback to english if detection fails (e.g. very short input)
            print("[RAG] Failed to detect language, falling back to English")
            return "English"


    # ──────────────────────────────────────────────────────────────────────────
    # ── Auxiliar functions ────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    def _is_a_negative_answer(self, answer, docs):
        """Return True if the model's answer signals lack of information."""
        if not docs:
            return True
        answer_lower = answer.lower()
        return any(re.search(p, answer_lower) for p in cfg.no_info_patterns)
        

    def _evaluate_turn(self, question, answer, docs, eval_mode=False):
        """Evaluate answer with Ragas, return the evaluation results."""

        contexts = [doc.page_content for doc in docs]
        
        # Set up the mini-dataset for Ragas evaluation
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts]
        }
        dataset = Dataset.from_dict(data)
        
        try:
            results = evaluate(
                dataset=dataset,
                metrics=[faithfulness,answer_relevancy,ContextUtilization()],
                llm=self.eval_llm,
                embeddings=self.embeddings_model,
                show_progress= not eval_mode,   # disable progress bar
            )
            df = results.to_pandas()
            print(
                f"\n[Eval] Faithfulness={df['faithfulness'].iloc[0]:.3f} | "
                f"Relevancy={df['answer_relevancy'].iloc[0]:.3f} | "
                f"CtxUtil={df['context_utilization'].iloc[0]:.3f}"
            )
            self._stamp_context_docs(docs)
            return results
        except Exception as e:
            print(f"[_evaluate_turn]: Error in Ragas evaluation: {e}")
            return None


    def _update_doc_metrics(self, docs, faithfulness_score, relevancy_score, ctx_score):
        """
        Attribute the current turn's Ragas scores to every retrieved doc's source URL.
        This builds a per-document performance profile over many evaluation turns.
        """
        for doc in docs:
            url = doc.metadata.get("source", "unknown")
            if url not in self.doc_metrics:
                self.doc_metrics[url] = {
                    "title": doc.metadata.get("title", "unknown"),
                    "faithfulness": [],
                    "answer_relevancy": [],
                    "context_utilization": [],
                    "appearances": 0,
                }
            self.doc_metrics[url]["faithfulness"].append(faithfulness_score)
            self.doc_metrics[url]["answer_relevancy"].append(relevancy_score)
            self.doc_metrics[url]["context_utilization"].append(ctx_score)
            self.doc_metrics[url]["appearances"] += 1

    
    def get_doc_performance_summary(self) -> list[dict]:
        """
        Returns a sorted list of dicts with average Ragas metrics per source URL.
        Useful for identifying best/worst performing documents.
        """
        summary = []
        for url, data in self.doc_metrics.items():
            def _mean(lst):
                return round(sum(lst) / len(lst), 4) if lst else None
 
            summary.append({
                "source": url,
                "title": data["title"],
                "appearances": data["appearances"],
                "avg_faithfulness":        _mean(data["faithfulness"]),
                "avg_answer_relevancy":    _mean(data["answer_relevancy"]),
                "avg_context_utilization": _mean(data["context_utilization"]),
            })
 
        # Sort by avg_faithfulness descending
        summary.sort(key=lambda x: x["avg_faithfulness"] or 0, reverse=True)
        return summary
    

    def summarize_docs(self, docs, lang: str = "English") -> str:
        """
        Synthesizes information from multiple retrieved documents into a single coherent summary.
        """
        if not docs:
            return "No relevant information found in the database."
        
        # 1. Prepare the combined content for the model
        context_to_summarize = ""
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Document")
            context_to_summarize += f"--- Source {i}: {title} ---\n{doc.page_content}\n\n"
            
        # 2. Specialized synthesis prompt
        # We ask the model to act as an expert medical librarian
        prompt = f"""
        You are an expert medical librarian. Synthesize the following technical fragments 
        into a single, coherent summary of approximately 5 lines.
        
        Focus strictly on: 
        1. The core pediatric infection or condition discussed.
        2. Key symptoms or vaccines mentioned across all sources.
        3. The most important clinical takeaway or warning.

        The summary MUST be written entirely in {lang}. 
        Do not mention "Source 1 says..." or "Document 2 mentions...", create a unified flow.
        
        Context:
        {context_to_summarize}
        
        Summary:"""

        try:
            response = self.client.generate(
                model=cfg.model_name,
                prompt=prompt,
                options={
                    "num_predict": 250, 
                    "temperature": cfg.temperature,
                    "top_p": 0.9
                }
            )
            return response['response'].strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
        
    
    # ──────────────────────────────────────────────────────────────────────────
    # ── Display functions ─────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    def _stamp_context_docs(self, docs):
        print(f"\n[RAGChat] 🎯 Chunks retrieved: {len(docs)}")
        print("📄 Sources used (ordered by semantic relevance):")
 
        seen_sources = {}
        for doc in docs:
            url   = doc.metadata.get("source", "unknown")
            title = doc.metadata.get("title", "unknown")
            score = doc.metadata.get("relevance_score", 0.0)
            if url not in seen_sources:
                seen_sources[url] = {"title": title, "max_score": score, "chunk_count": 1}
            else:
                seen_sources[url]["chunk_count"] += 1
 
        for url, info in seen_sources.items():
            chunks_str = f"{info['chunk_count']} chunk{'s' if info['chunk_count']>1 else ''}"
            print(f"  ▪ [Score: {info['max_score']:.3f}] {info['title']} ({chunks_str})")
            print(f"    🔗 {url}")
 
 
    def _stamp_and_store_metrics(self, generation_time, eval_results, negative_phrase):
        """Store and display per-turn metrics."""
        self.metrics_history["response_times"].append(generation_time)
        self.metrics_history["coverage"].append(0 if negative_phrase else 1)
 
        if not negative_phrase and eval_results is not None:
            df = eval_results.to_pandas()
            self.metrics_history["faithfulness"].append(df["faithfulness"].iloc[0])
            self.metrics_history["relevance"].append(df["answer_relevancy"].iloc[0])
            self.metrics_history["context_util"].append(df["context_utilization"].iloc[0])
 
        h = self.metrics_history
        def safe_mean(lst): return statistics.mean(lst) if lst else 0
 
        avg_time  = safe_mean(h["response_times"])
        avg_cov   = safe_mean(h["coverage"]) * 100
        avg_faith = safe_mean(h["faithfulness"])
        avg_rel   = safe_mean(h["relevance"])
        avg_ctx   = safe_mean(h["context_util"])
        avg_q     = safe_mean([f+r+c for f,r,c in zip(h["faithfulness"], h["relevance"], h["context_util"])]) / 3 if h["faithfulness"] else 0
 
        current_cov = 0 if negative_phrase else 100
        if not negative_phrase and h["faithfulness"]:
            cf  = h["faithfulness"][-1]
            cr  = h["relevance"][-1]
            cc  = h["context_util"][-1]
            cq  = (cf + cr + cc) / 3
        else:
            cf = cr = cc = cq = "N/A"
 
        print("\n" + "─"*80)
        print("|" + " "*24 + " 📊 REAL-TIME METRICS " + " "*24 + "|")
        print("─"*80)
        print(f"| {'Research Criterion':<25} | {'Current Turn':<22} | {'Session Average':<23} |")
        print("─"*80)
        print(f"| {'⏱️ Response Time':<25} | {generation_time:>18.3f} s | {avg_time:>21.3f} s |")
        print(f"| {'📚 Document Coverage':<25} | {current_cov:>19.1f} % | {avg_cov:>21.1f} % |")
        if isinstance(cq, str):
            print(f"| {'🧠 RAG System Quality':<25} | {cq:>20} | {avg_q:>21.3f}   |")
        else:
            print(f"| {'🧠 RAG System Quality':<25} | {cq:>20.3f}   | {avg_q:>21.3f}   |")
            print(f"|   ├─ Faithfulness           | {cf:>20.3f}   | {avg_faith:>21.3f}   |")
            print(f"|   ├─ Answer Relevancy       | {cr:>20.3f}   | {avg_rel:>21.3f}   |")
            print(f"|   └─ Context Utilization    | {cc:>20.3f}   | {avg_ctx:>21.3f}   |")
        print("─"*80)
 
 
    def _stamp_eval_metrics(self, results_data, total_time, TP, FP, TN, FN):
        """Print final evaluation summary for batch testing."""
        total    = TP + FP + TN + FN
        answered = TP + FP
 
        def safe_mean(lst):
            clean = []
            for x in lst:
                try:
                    if hasattr(x, "iloc"): x = x.iloc[0]
                    x = float(x)
                    if not np.isnan(x): clean.append(x)
                except Exception:
                    continue
            return sum(clean) / len(clean) if clean else 0.0
 
        avg_time    = total_time / max(total, 1)
        coverage_pct = (answered / total) * 100 if total > 0 else 0
        faith = safe_mean(results_data.get("faithfulness", []))
        rel   = safe_mean(results_data.get("relevancy", []))
        ctx   = safe_mean(results_data.get("context_util", []))
        sys_quality = (faith + rel + ctx) / 3
 
        print("\n" + "═"*70)
        print("📊 RESEARCH CRITERIA: RAG EVALUATION SUMMARY".center(70))
        print("═"*70)
        print(f"{'⏱️  Response Time (Speed)':<35}: {avg_time:>10.3f} s / query")
        print(f"{'📚 Document Coverage':<35}: {coverage_pct:>10.1f} % ({answered}/{total} answered)")
        print(f"{'🧠 RAG System Quality (Overall)':<35}: {sys_quality:>10.3f} / 1.000")
        print(f"   ├─ Faithfulness (Accuracy):      {faith:>10.3f}")
        print(f"   ├─ Answer Relevancy (Utility):   {rel:>10.3f}")
        print(f"   └─ Context Util. (Coherence):    {ctx:>10.3f}")
 
        accuracy  = (TP + TN) / total if total else 0
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall    = TP / (TP + FN) if (TP + FN) else 0
        f1        = (2*precision*recall/(precision+recall)) if (precision+recall) else 0
        halluc    = FP / (FP + TN) if (FP + TN) else 0
        miss      = FN / (TP + FN) if (TP + FN) else 0
 
        print("\n" + "═"*70)
        print("📉 EXTENDED CLASSIFICATION METRICS".center(70))
        print("═"*70)
        print(f"{'':<20} {'Pred=1 (Answered)':>18} {'Pred=0 (Refused)':>18}")
        print(f"{'Actual=1 (In DB)':<20} {TP:>18} {FN:>18}")
        print(f"{'Actual=0 (Not in DB)':<20} {FP:>18} {TN:>18}")
        print(f"\n{'Accuracy':<35}: {accuracy:>10.3f}")
        print(f"{'Precision':<35}: {precision:>10.3f}")
        print(f"{'Recall':<35}: {recall:>10.3f}")
        print(f"{'F1 Score':<35}: {f1:>10.3f}")
        print(f"{'Hallucination rate (FP)':<35}: {halluc:>10.3f}")
        print(f"{'Miss rate (FN)':<35}: {miss:>10.3f}")
        print("═"*70)


#───────────────────────────────────────────────────────────────────────────────
# ─── Auxiliary: Custom Reranker ───────────────────────────────────────────────
#───────────────────────────────────────────────────────────────────────────────

class CustomCrossEncoderReranker(CrossEncoderReranker):
    """Override LangChain's compressor to persist relevance scores in metadata."""
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
 
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        docs_with_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
 
        return [
            Document(
                page_content=doc.page_content,
                metadata={**doc.metadata, "relevance_score": float(score)}
            )
            for doc, score in docs_with_scores[:self.top_n]
        ]
 


def _normalise_field(value) -> str:
    """
    Convert any JSON value (str, list, dict, number) to a clean human-readable string.
 
    The model sometimes returns:
      - "answer": ["DTaP", "Hib", "IPV"]          → bullet list
      - "reasoning": {"assess_evidence": "...", …}  → labelled paragraphs
      - "answer": "plain string"                    → returned as-is
    """
    if isinstance(value, str):
        return value.strip()
 
    if isinstance(value, list):
        # Flatten mixed lists: each item can itself be a str or dict
        lines = []
        for item in value:
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"• {k}: {v}")
            else:
                lines.append(f"• {item}")
        return "\n".join(lines)
 
    if isinstance(value, dict):
        # Convert dict keys to readable labels (snake_case → Title Case)
        lines = []
        for k, v in value.items():
            label = k.replace("_", " ").title()
            # Recursively normalise nested values
            body  = _normalise_field(v)
            lines.append(f"**{label}:** {body}")
        return "\n".join(lines)
 
    # Fallback for numbers, booleans, None
    return str(value) if value is not None else ""
 
 
def _parse_cot_response(raw: str) -> dict:
    """
    Robustly parse the LLM's JSON output.
 
    Handles:
    - Clean JSON with flat string values
    - JSON where "reasoning" is a nested dict and/or "answer" is a list
    - JSON wrapped in markdown fences
    - Partial JSON recoverable via regex
    - Completely unparseable output (last-resort fallback)
    """
    # ── 1. Strip markdown fences ──────────────────────────────────────────
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$",          "", cleaned)
        cleaned = cleaned.strip()
 
    # ── 2. Try full JSON parse ────────────────────────────────────────────
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            raw_reasoning = parsed.get("reasoning", "")
            raw_answer    = parsed.get("answer",    "")
 
            reasoning = _normalise_field(raw_reasoning)
            answer    = _normalise_field(raw_answer)
 
            if answer:
                return {"reasoning": reasoning, "answer": answer}
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
 
    # ── 3. Attempt to salvage a truncated JSON object ─────────────────────
    # Some models cut off mid-stream; try closing the brace and re-parsing.
    if cleaned.startswith("{") and not cleaned.endswith("}"):
        try:
            salvaged = json.loads(cleaned + "}")
            raw_reasoning = salvaged.get("reasoning", "")
            raw_answer    = salvaged.get("answer",    "")
            reasoning = _normalise_field(raw_reasoning)
            answer    = _normalise_field(raw_answer)
            if answer:
                return {"reasoning": reasoning, "answer": answer}
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
 
    # ── 4. Regex fallback (flat string values only) ───────────────────────
    reasoning_match = re.search(
        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL
    )
    answer_match = re.search(
        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', cleaned, re.DOTALL
    )
    reasoning = reasoning_match.group(1).replace("\\n", "\n") if reasoning_match else ""
    answer    = answer_match.group(1).replace("\\n", "\n")    if answer_match    else ""
 
    if answer:
        return {"reasoning": reasoning, "answer": answer}
 
    # ── 5. Last resort ────────────────────────────────────────────────────
    return {
        "reasoning": "[Reasoning unavailable — model did not return valid JSON]",
        "answer":    raw.strip(),
    }
