# Baseline Resume Screening RAG

This project is an intentionally unsecured baseline for measuring how retrieved resume text reaches a language model. It does not filter, sanitize, classify, or remove any resume content.

## Layout

Put resume PDFs in `data/`. The current supplied PDFs are in `resume_dataset/`, which is accepted as a compatibility fallback when `data/` does not exist. Every `*.pdf` found in the selected directory is discovered automatically.

The ingestion flow is:

```text
one PDF -> load all pages -> one LangChain Document -> one embedding -> persistent ChromaDB
```

No text splitter is used. The Chroma ID is the PDF filename stem, so rerunning ingestion updates the same resume identity rather than creating a new ID.

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The embedding model runs locally for Chroma retrieval. The recommendation LLM is called through the Hugging Face API; no LLM weights are downloaded locally. Create `.env` with:

```text
HUGGINGFACEHUB_API_TOKEN=your_hugging_face_token
```

The default hosted model is `Qwen/Qwen2.5-7B-Instruct`. Set `LLM_MODEL` in `.env` or pass `--llm-model` to use another model supported by your Hugging Face provider.

## Run ingestion

```powershell
py ingest.py
```

This creates the persistent `chroma_db/` directory. Run ingestion again after adding, changing, or removing PDFs. To experiment with removal, remove the PDF from the source directory and use a fresh `chroma_db/` directory so old records are not retained.

## Run retrieval and recommendation

```powershell
py rag.py "Find candidates with Python, Machine Learning and Django experience." --k 5
py rag.py "Find candidates with Java and Spring Boot." --k 5
py rag.py "Find candidates with React and Node.js." --k 5
py rag.py "Find the best Python developer." --k 5
```

The script first prints the company requirement, rank, candidate, resume ID, source filename, distance, and complete resume content for every retrieved result. Only after that display does it build the context and invoke the LLM. The final section prints the LLM-generated candidate recommendation.

## Baseline comparison

1. Ingest all PDFs and run the same query with the same `--k`.
2. Record which source filename contains the simulated instruction, its retrieval rank and distance, and the final recommendation.
3. Remove that PDF from the source directory.
4. Delete or rename `chroma_db/`, ingest again, and rerun the identical query.
5. Compare the retrieved candidates and final recommendation.

The retrieval transcript is the evidence that a resume reached the LLM context. Comparing the two final outputs shows whether the retrieved content changed the recommendation. The system itself does not label the resume or treat it differently from other candidate data.