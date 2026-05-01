import os, sys, time, re, statistics
import ollama
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langdetect import detect, LangDetectException
from ragas import evaluate
from ragas.metrics import (
    faithfulness,       # checks if the answer is supported by the retrieved context
    answer_relevancy,   # checks if the answer is relevant to the question
    ContextUtilization  # checks if the context is relevant to the question
    )
from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from utils import GenConfig as cfg
from utils import VectorDBFactory

class RAGChat:
    """
    Class to encapsulate the RAG chat logic, useful for both terminal 
    and Streamlit interfaces.

    --- 
    Attributes:
        - vectorDB: Chroma vector daabase instance if not provided, it will be created on initialization
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
            temperature=0       # for evaluation, we want deterministic responses
        )
        logging.getLogger("ragas").setLevel(logging.ERROR)

        self.embeddings_model = HuggingFaceEmbeddings(model_name=cfg.embedding_model)


        # 3. get the vector database ready, if not provided, create it
        print(f"[RAGChat] Loading vector database from {cfg.chroma_dir}...")
        try:
            vectorDB = VectorDBFactory(cfg.chroma_dir).get_vectorDB()
        except Exception as e:
            print(f"[RagChat] Error loading database: {e}")
            sys.exit(1)
        print(f"[RAGChat] Vector database done...")
        

        self.vectorDB = vectorDB

        # to store the conversation for coherence in the conversation
        self.chat_history = {
            "questions": [],
            "answers": [],
            "retrieved_docs": []
        }
        self.metrics_history = {
            "response_times": [],
            "coverage": [],
            "faithfulness": [],
            "relevance": [],
            "context_util": []
        }
    

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
                answer, docs = self._get_ai_response(user_input, lang)
                generation_time = time.time() - start_time
                
                print(f"\nAI: {answer}")

                # 3. Check if the answer is negative
                negative_phrase = self._is_a_negative_answer(answer, docs)

                # 4. Evaluate turn with Ragas (only if the answer is not negative)
                if not negative_phrase:
                    # add the disclamer in the appropiate language
                    print("\n"+cfg.disclamer_prompt[lang])
                    if eval_mode:
                        eval_results = self._evaluate_turn(user_input, answer, docs)

                # 5. Print the docs used for the answer (for traceability)
                # ── Stamp docs coverage if we have docs ─────────────────────────────────────────────────
                if not negative_phrase and docs:
                    print(f"[RAGChat]: Chunks recovered: {len(docs)}")
                    print(f"\n📄 Sources used:")
                    seen = set()
                    for doc in docs:
                        url = doc.metadata.get("source", "unknown")
                        title = doc.metadata.get("title", "unknown")
                        if url not in seen:
                            print(f"  - {title}: {url}")
                            seen.add(url)


                # 6. Show and store the metrics
                if eval_mode:
                    self._stamp_and_store_metrics(
                        generation_time, eval_results, negative_phrase)
                
                # Store the answer for future context in the conversation
                self.chat_history["questions"].append(user_input)
                self.chat_history["answers"].append(answer)
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
            - results_data: dictionary with the results.
        """

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
            try:
                q = item["question"]
                label = item["label"]
                start_time = time.time()

                # 1. language detection
                lang = self._detect_language(q)

                # 2. RAG pipeline
                answer, docs = self._get_ai_response(q, lang, indep_quest=True)
                elapsed = time.time() - start_time

                # 3. skip negatives
                is_negative = self._is_a_negative_answer(answer, docs)
                pred = 0 if is_negative else 1  # 0 = negative, 1 = positive
                # update the confusion matrix
                if label == 1:
                    TP += 1 if pred == 1 else 0
                    FN += 1 if pred == 0 else 0
                else:
                    FP += 1 if pred == 1 else 0
                    TN += 1 if pred == 0 else 0
                
                if is_negative:
                    continue

                # 4. evaluation
                eval_results = self._evaluate_turn(q, answer, docs, eval_mode=True)
                total_time += elapsed

                if eval_results is None:
                    continue

                df = eval_results.to_pandas()

                results_data["question"].append(q)
                results_data["answer"].append(answer)
                results_data["contexts"].append([d.page_content for d in docs])

                results_data["faithfulness"].append(df["faithfulness"][0])
                results_data["relevancy"].append(df["answer_relevancy"][0])
                results_data["context_util"].append(df["context_utilization"][0])

            except Exception as e:
                print(f"[eval_questions] Error processing question: {q}\n{e}")

        # ── Final dataset ────────────────────────────────────────────────
        dataset = Dataset.from_dict(results_data)

        # ── Stamp summary ─────────────────────────────────────────────────────
        self._stamp_eval_metrics(
            dataset,            # eval metrics
            total_time,         # total evaluation time
            TP, FP, TN, FN,     # confusion matrix
        )

        return dataset


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


    def _get_ai_response(
            self, user_input, lang: str = "English", indep_quest: bool = False):
        """
        Implement the loop of RAG.
        1. Retrieve relevant chunks from the vectorDB
        2. Construct the prompt with the retrieved context and the user question
        3. Call the Ollama API with the constructed prompt and return the response

        ---
        Attributes:
            - user_input: the question from the user
            - lang: the language of the user input
            - indep_quest: if True, the question is treated as independent
        """

        # 1. Retrieval
        docs = self.vectorDB.similarity_search(user_input, k=cfg.retrieval_num)  # n best chunks

        # 1.1 Add source info to the context for better traceability:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            title = doc.metadata.get("title", "Unknown")
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"[Source {i}: {title} | {source}]\n{doc.page_content}")
        context = "\n\n".join(context_parts)

        # 1.2 Add the context of previous conversation turns
        if not indep_quest:
            prev_conversation = ""
            for q, a in zip(self.chat_history['questions'], self.chat_history['answers']):
                    prev_conversation += f"Parent: {q}\nAI: {a}\n\n"
            context = f"Previous conversation: {prev_conversation}\n\n" + context

        # 2. Build the prompt for the model, we include instructions and context
        final_prompt = f"""You are a professional Pediatric Assistant. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say you don't know.
        If a different topic question is posed (example: "What is the weather today?", "What can I have for breakfast or for lunch?", "What is your favourite color?", etc.) you should answer that you don't have that information.
        Never invent any response; if there is not in the data, answer that you don't know.

        CRITICAL LANGUAGE RULE: The user is writing in {lang}.
        You MUST respond entirely in {lang}. No exceptions.
        Example: if {lang} is Spanish, write the full answer in Spanish.
        Even the disclaimer at the end must be in {lang}.

        Context: {context}

        Question: {user_input}
        Helpful Answer in {lang}:"""

        # 3. call the model for a response
        response = self.client.chat(
            model=cfg.model_name,
            messages=[{
                "role": "user",
                "content": final_prompt,
            }]
        )
        
        return response['message']['content'], docs
    

    # ──────────────────────────────────────────────────────────────────────────
    # ── Auxiliar functions ────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────────

    def _is_a_negative_answer(self, answer, docs):
        """
        Check if the answer is a negative one where model is saying it 
        doesn't know or doesn't have information.
        """
        # Search for pattens in all the answer
        answer_lower = answer.lower()

        # if there are no docs retrieved answer is negative.
        if not docs:
            return True
        
        # check for negative patterns in the answer
        has_no_info = any(re.search(pattern, answer_lower) for pattern in cfg.no_info_patterns)
        return has_no_info


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
                metrics=[
                    faithfulness, 
                    answer_relevancy, 
                    ContextUtilization()],      # no gold answers metrics
                llm=self.eval_llm,
                embeddings=self.embeddings_model,
                show_progress= not eval_mode,   # disable progress bar
            )
            print(f"\n[RAGChat] Evaluation results:\n{results}")
            return results
        except Exception as e:
            print(f"[_evaluate_turn]: Error in Ragas evaluation: {e}")
            return None


    def _stamp_and_store_metrics(self, generation_time, eval_results, negative_phrase):
        """Store and stamp the metrics of the turn"""
        
        # ── Store the metrics in the history for the final report ─────────────
        self.metrics_history["response_times"].append(generation_time)

        if not negative_phrase and eval_results is not None:
            df = eval_results.to_pandas()
            self.metrics_history["faithfulness"].append(df['faithfulness'].iloc[0])
            self.metrics_history["relevance"].append(df['answer_relevancy'].iloc[0])
            self.metrics_history["context_util"].append(df['context_utilization'].iloc[0])

        

        # ── Stamp the metrics ─────────────────────────────────────────────────
        h = self.metrics_history
        def safe_mean(lst):
            return statistics.mean(lst) if len(lst) > 0 else 0
        avg_time = safe_mean(h["response_times"])
        avg_faith = safe_mean(h["faithfulness"])
        avg_rel = safe_mean(h["relevance"])
        avg_context_util = safe_mean(h["context_util"])
        if not negative_phrase and len(h["faithfulness"]) > 0:
            current_faith = h["faithfulness"][-1]
            current_rel = h["relevance"][-1]
            current_context_util = h["context_util"][-1]
        else:
            current_faith = "N/A"
            current_rel = "N/A"
            current_context_util = "N/A"

        print("─"*57)
        print("|" + " "*22 + " 📊 METRICS " + " "*22 + "|")
        print("─"*57)
        print(f"|\tMetric             \t|\tCurrent\t|\tAverage'\t|")

        print("─"*57)
        print(f"|  ⏱️ Response time   \t|\t{generation_time:.3f}\t|\t{avg_time:.3f}\t|")
        print(f"|  ✅ Faithfulness    \t|\t{current_faith:.3f}\t|\t{avg_faith:.3f}\t|")
        print(f"|  🎯 Relevance       \t|\t{current_rel:.3f}\t|\t{avg_rel:.3f}\t|")
        print(f"|  📊 Context Precision\t|\t{current_context_util:.3f}\t|\t{avg_context_util:.3f}\t|")
        print("─"*57)


    def _stamp_eval_metrics(self, results_data, total_time, TP, FP, TN, FN):
        total = TP + FP + TN + FN
        accuracy = (TP + TN) / total if total > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        print("\n" + "═"*70)
        print("📊 BATCH EVALUATION SUMMARY".center(70))
        print("═"*70)

        avg_time = total_time / max(total, 1)

        print(f"{'Total questions':<30}: {total:>10}")
        print(f"{'Avg time / question':<30}: {avg_time:>10.2f} s")

        if results_data["faithfulness"]:
            print(f"{'Avg faithfulness':<30}: {statistics.mean(results_data['faithfulness']):>10.3f}")
            print(f"{'Avg relevancy':<30}: {statistics.mean(results_data['relevancy']):>10.3f}")
            print(f"{'Avg context utilization':<30}: {statistics.mean(results_data['context_util']):>10.3f}")

        # ── Confusion Matrix ───────────────────────────────────────────
        print("\n" + "═"*70)
        print("📉 CONFUSION MATRIX".center(70))
        print("═"*70)

        print(f"{'':<20} {'Pred=1':>10} {'Pred=0':>10}")
        print(f"{'Actual=1':<20} {TP:>10} {FN:>10}")
        print(f"{'Actual=0':<20} {FP:>10} {TN:>10}")

        # ── Classification Metrics ─────────────────────────────────────
        print("\n" + "═"*70)
        print("📊 CLASSIFICATION METRICS".center(70))
        print("═"*70)

        print(f"{'Accuracy':<30}: {accuracy:>10.3f}")
        print(f"{'Precision':<30}: {precision:>10.3f}")
        print(f"{'Recall':<30}: {recall:>10.3f}")
        print(f"{'F1 Score':<30}: {f1:>10.3f}")

        # ── Extra insight (muy útil en RAG) ────────────────────────────
        hallucination_rate = FP / (FP + TN) if (FP + TN) > 0 else 0

        print("\n" + "═"*70)
        print("🧠 RAG-SPECIFIC METRICS".center(70))
        print("═"*70)

        print(f"{'Hallucination rate':<30}: {hallucination_rate:>10.3f}")
        print(f"{'Miss rate (FN)':<30}: {FN / (TP + FN) if (TP + FN) > 0 else 0:>10.3f}")

        print("═"*70)