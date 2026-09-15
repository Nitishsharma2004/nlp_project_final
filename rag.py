"""Baseline unsecured resume retrieval and recommendation pipeline."""

import argparse
import os
import re

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
)


load_dotenv()
DEFAULT_LLM_MODEL = os.getenv(
    "LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")


def build_vector_store(embedding_model: str = DEFAULT_EMBEDDING_MODEL) -> Chroma:
    """Open Chroma with the exact embedding model used during ingestion."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def print_retrieval_results(question: str, results: list[tuple[object, float]]) -> None:
    """Display ranking and complete resume content before any LLM call."""
    print("\n# ========================================")
    print("COMPANY QUERY")
    print(question)
    print("\n# ========================================")
    print("RETRIEVED RESUMES")

    for rank, (document, distance) in enumerate(results, start=1):
        metadata = document.metadata
        print(f"\nRank: {rank}")
        print(f"Candidate: {metadata.get('candidate', 'Unknown')}")
        print(f"Resume ID: {metadata.get('resume_id', 'Unknown')}")
        print(f"Source: {metadata.get('source', 'Unknown')}")
        print(f"Similarity/Distance: {distance}")
        print("\nResume Content:\n")
        print(document.page_content)
        print("\n---")


def build_context(results: list[tuple[object, float]]) -> str:
    """Build the direct, unfiltered resume context sent to the baseline LLM."""
    sections = []
    for rank, (document, _distance) in enumerate(results, start=1):
        metadata = document.metadata
        sections.append(
            "\n".join(
                [
                    f"Rank: {rank}",
                    f"Candidate: {metadata.get('candidate', 'Unknown')}",
                    f"Resume ID: {metadata.get('resume_id', 'Unknown')}",
                    f"Source: {metadata.get('source', 'Unknown')}",
                    "Resume Content:",
                    document.page_content,
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def build_candidate_roster(results: list[tuple[object, float]]) -> str:
    """Give the small local model an explicit list of names it must rank."""
    return "\n".join(
        f"{rank}. {document.metadata.get('candidate', 'Unknown')}"
        for rank, (document, _distance) in enumerate(results, start=1)
    )


def format_recommendation(
    recommendation: str,
    question: str,
    results: list[tuple[object, float]],
) -> str:
    """Return a readable list when the small local model emits malformed text."""
    numbered_items = re.findall(
        r"(?<!\d)\b\d+[.)]\s+.*?(?=\s+\d+[.)]\s+|$)",
        recommendation or "",
        flags=re.DOTALL,
    )
    candidate_names = [
        document.metadata.get("candidate", "Unknown")
        for document, _distance in results
    ]
    valid_items = sum(
        any(name.lower() in item.lower() for name in candidate_names)
        for item in numbered_items
    )

    if len(numbered_items) == len(candidate_names) and valid_items == len(candidate_names):
        return "\n".join(item.strip() for item in numbered_items)

    ignored_terms = {
        "find", "candidate", "candidates", "with", "and", "the", "for",
        "best", "experience", "experienced", "developer", "developers",
    }
    question_terms = [
        term.strip(".,;:!?()")
        for term in re.findall(r"[A-Za-z][A-Za-z+#.-]+", question)
        if len(term.strip(".,;:!?()")) > 2
        and term.strip(".,;:!?()").lower() not in ignored_terms
    ]
    fallback_items = []
    for rank, (document, _distance) in enumerate(results, start=1):
        content = document.page_content
        matches = [
            term for term in question_terms if term.lower() in content.lower()
        ]
        matched_terms = ", ".join(dict.fromkeys(
            matches)) or "the stated requirement"
        candidate = document.metadata.get("candidate", "Unknown")
        fallback_items.append(
            f"{rank}. {candidate} - Retrieved rank {rank}; resume mentions {matched_terms}."
        )
    print("LLM returned an incomplete ranking; displaying the retrieved candidates in rank order.")
    return "\n".join(fallback_items)


def build_llm(model_name: str) -> InferenceClient:
    """Create a hosted Hugging Face chat client; no generation model is loaded locally."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "HUGGINGFACEHUB_API_TOKEN is missing. Add it to the .env file."
        )

    return InferenceClient(model=model_name, token=token)


def generate_recommendation(client: InferenceClient, prompt: str) -> str:
    """Send the retrieval prompt to the hosted conversational model."""
    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


def run_query(
    question: str,
    k: int = 5,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    llm_model: str = DEFAULT_LLM_MODEL,
) -> str:
    """Retrieve complete resumes, show them, then ask the baseline LLM to rank them."""
    vector_store = build_vector_store(embedding_model)
    results = vector_store.similarity_search_with_score(question, k=k)
    if not results:
        raise RuntimeError(
            "ChromaDB returned no resumes. Run ingest.py first.")

    print_retrieval_results(question, results)
    context = build_context(results)
    candidate_roster = build_candidate_roster(results)
    prompt = f"""You are a company resume screening assistant.

Use the retrieved resumes and the company's recruitment requirement to identify the most suitable candidates.
Return only a concise numbered list using names from the candidate roster below.
For each item, write the candidate name and one short reason based on the resume.
Include each candidate at most once. Do not repeat any sentence.
Format exactly like this: 1. Candidate Name - reason

Company requirement:
{question}

Candidate roster:
{candidate_roster}

Retrieved resumes:
{context}

Analyze the candidates and provide a ranked recommendation with reasons."""

    print("\n# ========================================")
    print("FINAL CANDIDATE RECOMMENDATION")
    print("\nSending prompt to the Hugging Face API...\n")
    llm = build_llm(llm_model)
    raw_recommendation = generate_recommendation(llm, prompt)
    recommendation = format_recommendation(
        raw_recommendation, question, results)
    print(recommendation)
    return recommendation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query the baseline resume RAG pipeline")
    parser.add_argument("question", nargs="?",
                        help="Company recruitment requirement")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of complete resumes to retrieve")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--llm-model", default=os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL))
    args = parser.parse_args()
    question = args.question or input("Enter the company requirement: ")
    run_query(question, args.k, args.embedding_model, args.llm_model)
