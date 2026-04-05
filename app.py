import streamlit as st
from groq import Groq
from rag_engine import get_pdf_text, split_into_chunks, create_vectorstore, search

st.title("📚 PDF Study Assistant")

groq = Groq(api_key=st.secrets["GROQ_API_KEY"])

pdf = st.file_uploader("Upload a PDF", type="pdf")

if pdf:
    with st.spinner("Reading PDF..."):
        text = get_pdf_text(pdf)
        chunks = split_into_chunks(text)
        db = create_vectorstore(chunks)
    st.success(f"Done! {len(chunks)} chunks indexed.")

    question = st.text_input("Ask something about the PDF:")

    if question:
        context = "\n".join(search(question, db))

        reply = groq.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": f"Answer using this context:\n{context}\n\nQuestion: {question}"}]
        )

        st.markdown("### Answer")
        st.write(reply.choices[0].message.content)