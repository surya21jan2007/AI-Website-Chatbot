import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from gpt4all import GPT4All

st.set_page_config(
    page_title="AI Website Chatbot",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 AI Website Chatbot")
st.write("Enter a website URL and ask questions about its content.")

url = st.text_input(
    "Enter Website URL",
    placeholder="https://example.com"
)

if url:

    with st.spinner("Loading Website..."):

        try:

            loader = WebBaseLoader(url)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            retriever = vectorstore.as_retriever(
                search_kwargs={"k": 3}
            )

            st.success("✅ Website Processed Successfully")

            question = st.text_input(
                "Ask a Question"
            )

            if question:

                retrieved_docs = retriever.invoke(question)

                context = "\n".join(
                    [doc.page_content for doc in retrieved_docs]
                )

                prompt = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
You are a helpful AI assistant.

Answer only from the provided website content.

If the answer is not found in the context,
say:
"I couldn't find that information on the website."

Context:
{context}

Question:
{question}

Answer:
"""
                )

                final_prompt = prompt.format(
                    context=context,
                    question=question
                )

                with st.spinner("Generating Answer..."):

                    llm = GPT4All(
                        "mistral-7b-instruct-v0.1.Q4_0.gguf"
                    )

                    answer = llm.generate(
                        final_prompt
                    )

                st.subheader("💡 Answer")
                st.write(answer)

                st.subheader("📄 Sources")

                for doc in retrieved_docs:
                    st.write(
                        doc.metadata.get(
                            "source",
                            "Website Content"
                        )
                    )

        except Exception as e:
            st.error(
                f"Error: {str(e)}"
            )