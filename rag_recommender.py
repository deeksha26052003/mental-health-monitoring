# rag_recommender.py
#
# Real RAG pipeline: embeds a small local knowledge base with OpenAI
# embeddings, retrieves the most relevant snippets for a given person's
# risk profile using cosine similarity, and asks an LLM to turn those
# snippets into one short, personalized, non-diagnostic suggestion.
#
# Requires:
#   pip install openai numpy --break-system-packages   (if on Linux)
#   pip install openai numpy                           (on Mac)
#
# Requires an OpenAI API key set as an environment variable:
#   export OPENAI_API_KEY="sk-..."          (Mac/Linux, in terminal)
#   $env:OPENAI_API_KEY="sk-..."            (Windows PowerShell)
#
# Note: this used to use faiss for retrieval, but faiss is a compiled C
# extension that conflicts with Tkinter's native macOS window toolkit and
# crashes the whole app with a segmentation fault. At this knowledge-base
# size (10 documents), plain NumPy cosine similarity is just as real a
# retrieval method and has no such conflict.

import os
import numpy as np
from openai import OpenAI

from knowledge_base import DOCUMENTS

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

client = OpenAI()  # reads OPENAI_API_KEY from environment automatically


def _embed_texts(texts):
    """Call OpenAI embeddings API for a list of texts, return np.array."""
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def _cosine_similarity(query_vec, doc_vecs):
    """Cosine similarity between one query vector and many document vectors."""
    query_norm = query_vec / np.linalg.norm(query_vec)
    doc_norms = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return doc_norms @ query_norm


class KnowledgeBaseIndex:
    """Embeds DOCUMENTS once, reuses the vectors for every query."""

    def __init__(self):
        self.tags = [tag for tag, _ in DOCUMENTS]
        self.texts = [text for _, text in DOCUMENTS]
        print("Embedding knowledge base (one-time setup)...")
        self.vectors = _embed_texts(self.texts)
        print(f"Indexed {len(self.texts)} documents.")

    def retrieve(self, query, k=3):
        """Return the top-k most relevant (tag, text) pairs for a query."""
        query_vector = _embed_texts([query])[0]
        similarities = _cosine_similarity(query_vector, self.vectors)
        top_k_indices = np.argsort(similarities)[::-1][:k]
        results = [(self.tags[i], self.texts[i]) for i in top_k_indices]
        return results


def build_query_from_risk_factors(user_data, prediction, probability):
    """
    Turn the structured inputs already used by predict.py into a short
    natural-language query describing this person's risk profile, so
    retrieval can find the most relevant coping-strategy snippets.
    """
    factors = []

    if user_data.get("Do you have Anxiety?") == 1:
        factors.append("anxiety")
    if user_data.get("Do you have Panic attack?") == 1:
        factors.append("panic attacks")
    if user_data.get("Sleep Schedule") == "Irregular" or user_data.get("Sleep Schedule") == 0:
        factors.append("irregular sleep schedule")
    if user_data.get("Exercise Frequency") in ("Never", 0):
        factors.append("little to no exercise")
    if user_data.get("Family History of Mental Illness") == 1:
        factors.append("family history of mental illness")
    if user_data.get("Substance Use") in ("Occasional", "Frequent", 1, 2):
        factors.append("substance use as a coping mechanism")
    if user_data.get("Work-Life Balance", 10) <= 4:
        factors.append("poor work-life balance")
    if user_data.get("Daily Screen Time (hours)", 0) >= 8:
        factors.append("high daily screen time")

    factor_str = ", ".join(
        factors) if factors else "no major elevated risk factors"

    query = (
        f"Depression risk prediction: {'elevated' if prediction == 1 else 'low'} "
        f"(confidence {probability * 100:.0f}%). "
        f"Relevant risk factors present: {factor_str}."
    )
    return query, factors


def generate_recommendation(kb_index, user_data, prediction, probability):
    """
    Full RAG call: build query -> retrieve top-k docs -> generate a short,
    grounded, non-diagnostic suggestion using only the retrieved context.
    """
    query, factors = build_query_from_risk_factors(
        user_data, prediction, probability)

    retrieved = kb_index.retrieve(query, k=3)

    # Safety: always include the crisis resource snippet if risk is elevated,
    # regardless of whether it was the top-3 retrieval match.
    if prediction == 1 and probability >= 0.6:
        crisis_doc = next((t for tag, t in DOCUMENTS if tag == "crisis"), None)
        if crisis_doc and crisis_doc not in [text for _, text in retrieved]:
            retrieved.append(("crisis", crisis_doc))

    context_block = "\n\n".join(f"- {text}" for _, text in retrieved)

    system_prompt = (
        "You are a supportive assistant that turns short evidence snippets "
        "into ONE brief, warm, practical suggestion for someone who just "
        "received a screening result from a machine learning model. "
        "Rules:\n"
        "1. You are not a therapist and must not diagnose. Frame this as a "
        "screening signal, not a diagnosis.\n"
        "2. Base your suggestion ONLY on the provided context snippets. Do "
        "not invent medical claims beyond them.\n"
        "3. Keep it to 3-5 sentences, warm and plain-language.\n"
        "4. If a crisis-resource snippet is present in the context, include "
        "that resource clearly and verbatim near the end.\n"
        "5. Do not use clinical jargon."
    )

    user_prompt = (
        f"Screening result: {query}\n\n"
        f"Context snippets to base your suggestion on:\n{context_block}\n\n"
        "Write the suggestion now."
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
        max_tokens=250,
    )

    suggestion = response.choices[0].message.content.strip()
    return suggestion, retrieved, factors


if __name__ == "__main__":
    # Quick manual test — run this file directly to sanity check the pipeline
    # before wiring it into predict.py.
    kb_index = KnowledgeBaseIndex()

    fake_user_data = {
        "Do you have Anxiety?": 1,
        "Do you have Panic attack?": 0,
        "Sleep Schedule": "Irregular",
        "Exercise Frequency": "Never",
        "Family History of Mental Illness": 0,
        "Substance Use": "Occasional",
        "Work-Life Balance": 3,
        "Daily Screen Time (hours)": 9.5,
    }
    fake_prediction = 1
    fake_probability = 0.72

    suggestion, retrieved, factors = generate_recommendation(
        kb_index, fake_user_data, fake_prediction, fake_probability
    )

    print("\n--- Retrieved snippets ---")
    for tag, text in retrieved:
        print(f"[{tag}] {text[:80]}...")

    print("\n--- Generated suggestion ---")
    print(suggestion)
