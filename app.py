import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from rag_engine import DocumentRAG

load_dotenv()
st.set_page_config(page_title="Chat With Your Documents", page_icon="📚", layout="wide")

st.title("📚 Chat With Your Documents")
st.caption("Upload documents, retrieve relevant passages, and ask questions grounded in your files.")

with st.sidebar:
    st.header("1. Add documents")
    uploads = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    process = st.button("Process documents", type="primary", use_container_width=True)
    st.divider()
    st.header("2. Settings")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your key is used only by this local app. It is not saved to the project.",
    )
    model = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"], index=0)
    st.caption("Embeddings run locally using sentence-transformers.")

@st.cache_resource
def get_rag():
    return DocumentRAG()

rag = get_rag()

if process:
    if not uploads:
        st.warning("Upload at least one document first.")
    else:
        with st.spinner("Extracting text, creating chunks and embeddings..."):
            docs = []
            for upload in uploads:
                docs.append((upload.name, upload.getvalue()))
            try:
                count = rag.add_documents(docs)
                st.success(f"Processed {len(docs)} file(s) into {count} searchable chunks.")
            except Exception as exc:
                st.error(f"Could not process documents: {exc}")

stats = rag.stats()
c1, c2, c3 = st.columns(3)
c1.metric("Indexed chunks", stats["chunks"])
c2.metric("Documents", stats["documents"])
c3.metric("Embedding model", "MiniLM-L6-v2")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"- **{source['file']}** — chunk {source['chunk']} (similarity: {source['score']:.3f})")

question = st.chat_input("Ask a question about your uploaded documents...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if stats["chunks"] == 0:
            answer = "Please upload and process at least one document before asking a question."
            sources = []
        elif not api_key:
            answer = "Add your OpenAI API key in the sidebar, then ask your question again."
            sources = []
        else:
            try:
                with st.spinner("Searching documents and generating an answer..."):
                    results = rag.retrieve(question, top_k=top_k)
                    if not results:
                        answer = "I couldn't find relevant information in the uploaded documents."
                        sources = []
                    else:
                        context = "\n\n".join(
                            f"[Source {i+1}: {r['file']}, chunk {r['chunk']}]\n{r['text']}"
                            for i, r in enumerate(results)
                        )
                        client = OpenAI(api_key=api_key)
                        response = client.chat.completions.create(
                            model=model,
                            temperature=0.1,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a document question-answering assistant. "
                                        "Answer using only the supplied context. If the answer is not "
                                        "supported by the context, say you cannot find it in the documents. "
                                        "Do not invent facts. Cite supporting sources inline as [Source 1], "
                                        "[Source 2], etc. Keep the answer clear and concise."
                                    ),
                                },
                                {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"},
                            ],
                        )
                        answer = response.choices[0].message.content or "No answer was returned."
                        sources = results
                st.markdown(answer)
                if sources:
                    with st.expander("Retrieved source passages"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(
                                f"**[Source {i}] {source['file']} — chunk {source['chunk']}**  \n"
                                f"Similarity: `{source['score']:.3f}`"
                            )
                            st.write(source["text"])
            except Exception as exc:
                answer = f"An error occurred: {exc}"
                sources = []
                st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
