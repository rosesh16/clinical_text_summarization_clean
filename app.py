import streamlit as st

from chasm_modules import CHASMSummarizer, FactualVerifier, verify_and_refine_summary
from rag_pipeline import RAGPipeline
from src.logger import logging
from utils import chunk_text, extract_text_from_pdf


DEFAULT_QUERY = "summarize key medical findings"


@st.cache_resource(show_spinner=False)
def load_rag_pipeline(embedding_model_name: str) -> RAGPipeline:
    return RAGPipeline(embedding_model_name=embedding_model_name)


@st.cache_resource(show_spinner=False)
def load_chasm_summarizer(model_name: str) -> CHASMSummarizer:
    return CHASMSummarizer(model_name=model_name)


@st.cache_resource(show_spinner=False)
def load_factual_verifier(model_name: str) -> FactualVerifier:
    return FactualVerifier(model_name=model_name)


def main():
    st.set_page_config(page_title="Medical Document Summarizer (RAG + CHASM)", layout="wide")
    st.title("Medical Document Summarizer (RAG + CHASM)")
    st.caption("Upload a medical PDF, retrieve the most relevant chunks, summarize them hierarchically, and verify the final answer.")

    with st.sidebar:
        st.header("Controls")
        top_k = st.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=3)
        chunk_min_words = st.slider("Chunk min words", min_value=200, max_value=400, value=300, step=25)
        chunk_max_words = st.slider("Chunk max words", min_value=350, max_value=700, value=500, step=25)
        chunk_summary_length = st.slider("Chunk summary max length", min_value=60, max_value=180, value=110, step=10)
        final_summary_length = st.slider("Final summary max length", min_value=100, max_value=300, value=180, step=10)
        verification_threshold = st.slider("Verification threshold", min_value=0.30, max_value=0.90, value=0.55, step=0.05)

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    query = st.text_input("Query", value=DEFAULT_QUERY, placeholder=DEFAULT_QUERY)
    generate = st.button("Generate Summary", type="primary")

    if not generate:
        return

    if uploaded_file is None:
        st.warning("Please upload a PDF file before generating a summary.")
        return

    try:
        with st.spinner("Extracting document text..."):
            document_text = extract_text_from_pdf(uploaded_file)

        if not document_text.strip():
            st.error("The uploaded PDF appears to be empty or text could not be extracted.")
            return

        with st.spinner("Building semantic chunks and vector index..."):
            chunks = chunk_text(document_text, min_words=chunk_min_words, max_words=chunk_max_words)

        if not chunks:
            st.error("No meaningful text chunks could be created from this document.")
            return

        rag_pipeline = load_rag_pipeline("sentence-transformers/all-MiniLM-L6-v2")
        summarizer = load_chasm_summarizer("sshleifer/distilbart-cnn-12-6")
        verifier = load_factual_verifier("facebook/bart-large-mnli")

        with st.spinner("Computing embeddings and retrieving relevant context..."):
            rag_pipeline.build_index(chunks)
            retrieval_result = rag_pipeline.retrieve(query=query.strip() or DEFAULT_QUERY, top_k=top_k)

        with st.spinner("Running multi-stage CHASM summarization..."):
            intermediate_summaries = summarizer.summarize_chunks(
                retrieval_result.retrieved_chunks,
                max_summary_length=chunk_summary_length,
                min_summary_length=max(20, chunk_summary_length // 3),
            )
            draft_summary = summarizer.combine_summaries(
                intermediate_summaries,
                max_summary_length=final_summary_length,
                min_summary_length=max(40, final_summary_length // 3),
            )

        with st.spinner("Verifying factual consistency..."):
            verification = verify_and_refine_summary(
                summarizer=summarizer,
                verifier=verifier,
                summary=draft_summary,
                evidence_chunks=retrieval_result.retrieved_chunks,
                threshold=verification_threshold,
                max_refinement_attempts=1,
                max_summary_length=final_summary_length,
                min_summary_length=max(40, final_summary_length // 3),
            )

        st.subheader("Retrieved Context")
        for idx, chunk in enumerate(retrieval_result.retrieved_chunks, start=1):
            score = retrieval_result.similarity_scores[idx - 1] if idx - 1 < len(retrieval_result.similarity_scores) else 0.0
            with st.expander(f"Chunk {idx} | Similarity: {score:.4f}", expanded=idx == 1):
                st.write(chunk)

        st.subheader("Intermediate Summaries")
        for idx, summary in enumerate(intermediate_summaries, start=1):
            st.markdown(f"**Chunk Summary {idx}**")
            st.write(summary)

        st.subheader("Final Summary")
        st.write(verification.final_summary)

        col1, col2 = st.columns(2)
        col1.metric("Factual Support Score", f"{verification.factual_score:.2f}")
        col2.metric("Refined After Verification", "Yes" if verification.refined else "No")

        st.subheader("Verification Details")
        for idx, result in enumerate(verification.claim_results, start=1):
            status = "Supported" if result["supported"] else "Needs review"
            st.markdown(
                f"**Claim {idx}**: {status} | "
                f"Entailment: {result['best_entailment']:.2f} | "
                f"Contradiction: {result['best_contradiction']:.2f}"
            )
            st.write(result["claim"])

        logging.info("Summary generated successfully for %s", uploaded_file.name)
    except Exception as exc:
        logging.exception("Application error while generating summary")
        st.error(f"Something went wrong while processing the document: {exc}")


if __name__ == "__main__":
    main()
