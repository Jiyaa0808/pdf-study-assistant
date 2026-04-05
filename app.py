import streamlit as st
from groq import Groq
from rag_engine import get_pdf_text, split_into_chunks, create_vectorstore, search

st.title("📚 PDF Study Assistant")
groq = Groq(api_key=st.secrets["GROQ_API_KEY"])

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    # Unique key for this file — only rebuild db if it's a NEW file
    file_id = pdf.name + str(pdf.size)

    if st.session_state.get("file_id") != file_id:
        with st.spinner("Reading PDF..."):
            text = get_pdf_text(pdf)
            chunks = split_into_chunks(text)
            db = create_vectorstore(chunks)
        st.session_state["file_id"] = file_id
        st.session_state["db"] = db
        st.session_state["chunks_count"] = len(chunks)

    db = st.session_state["db"]
    st.success(f"Done! {st.session_state['chunks_count']} chunks indexed.")

    question = st.text_input("Ask something about the PDF:")

    if question:
        results = search(question, db)
        if not results:
            st.warning("No relevant content found.")
        else:
            context = "\n".join(results)[:2000]
            reply = groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": f"Answer using this context:\n{context}\n\nQuestion: {question}"}]
            )
            st.markdown("### Answer")
            st.write(reply.choices[0].message.content)