import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.llms import Cohere
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set page config
st.set_page_config(page_title="Cohere RAG", layout="wide")

# Custom CSS to improve the UI and move sidebar to the left
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .stApp {
        max-width: 1500px;
        margin: 0 auto;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn.sanity.io/images/rjtqmwfu/production/5a374837aab376bb677b3a968c337532ea16f6cb-800x600.png?rect=0,90,800,420&w=1200&h=630", width=200)
    st.title("PDF Upload & Settings")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    cohere_api_key = st.text_input("Enter your Cohere API key", type="password")

    if uploaded_file:
        st.success("PDF uploaded successfully!")
    if cohere_api_key:
        st.success("API key entered!")

# Main content
st.title("Cohere RAG APP")
st.markdown("---")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

# Process PDF
if uploaded_file is not None and cohere_api_key and not st.session_state.pdf_processed:
    with st.spinner("Processing PDF... This may take a moment."):
        # Define Embeddings Model
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

        # Read PDF and extract text
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=['\n', '\n\n', ' ', '']
        )
        chunks = text_splitter.split_text(text=pdf_text)

        # Create vectorstore
        vectorstore = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="./chroma_db")
        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_processed = True
    
    st.success("PDF processed successfully!")

# Question input
question = st.text_input("What would you like to know about the PDF content?", placeholder="Enter your question here...")

# Generate answer
if st.session_state.pdf_processed and question:
    if st.button("Get Answer"):
        with st.spinner("Generating answer..."):
            # Define LLM
            cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key=cohere_api_key)

            vectorstore = st.session_state.vectorstore
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

            # Define prompt template
            prompt_template = """You are an expert assistant. Answer the question as accurately and precisely as possible using only the provided context.
                                If the answer is not contained in the context, respond with "answer not available in context."
                                Context:{context}
                                Question:{question}
                                Answer:"""
            
            prompt = PromptTemplate.from_template(template=prompt_template)

            # Format docs function
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # RAG Chain
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | cohere_llm
                | StrOutputParser()
            )

            answer = rag_chain.invoke(question)
            st.markdown("### Answer:")
            st.info(answer)

elif not st.session_state.pdf_processed:
    st.warning("Please upload a PDF file and enter your Cohere API key first.")
elif st.session_state.pdf_processed and not question:
    st.info("PDF processed. Please enter a question to get an answer.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Cohere.")
