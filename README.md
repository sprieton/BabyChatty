# 👶 BabyChatty - Pediatric RAG Assistant

This project is an intelligent assistant based on **RAG (Retrieval-Augmented Generation)** designed to answer pediatric health questions using medical data. The system utilizes **Llama 3.1** via the university infrastructure and a local vector database for context-aware responses.

## 🚀 Project Structure

- `src/`: Source code (Scraper, Ingestion, Chat Logic, and Configuration).
- `data/`: Retrieved documents in `.parquet` and `.jsonl` formats.
- `chroma_db/`: Persistent vector database (automatically generated if missing).
- `main.py`: Main entry point to launch the application.
- `requirements.txt`: List of Python dependencies.

## 🛠️ Installation and Setup

Follow these steps to replicate the development environment on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/sprieton/BabyChatty.git](https://github.com/sprieton/BabyChatty.git)
cd BabyChatty
```

### 2. Create and Activate the Virtual Environment
The virtual environment for this project is specifically named baby_venv.

On Windows (PowerShell):
```bash
python -m venv baby_venv
.\baby_venv\Scripts\activate
```

On Linux / Mac:
```bash
python3 -m venv baby_venv
source baby_venv/bin/activate
```

### 3. Install Dependencies
Once the baby_venv is activated, install all necessary libraries:
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a .env file in the root directory and add your API credentials:
```bash
OLLAMA_API_KEY=your_key_here
```

📈 System Usage
To start the assistant, simply run the main script:
```bash
python main.py
```

Note: On the first run, the system will detect the absence of the vector database and automatically perform data ingestion from the files located in the data/ folder.

📝 Development Notes
Chunking: The system uses recursive character splitting (1000 chars with 150 overlap) to preserve medical context.

Embeddings: Powered by all-MiniLM-L6-v2 for a balance between speed and retrieval accuracy.

Frontend: A Streamlit interface is currently under development to replace the CLI.