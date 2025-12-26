import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


load_dotenv()

st.title("PDF Question Answering with Google Gemini")

loader = PyPDFLoader("attentionisallyouneedgemini.pdf")
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma.from_documents(documents=docs, embedding = embeddings)


retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":10})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3,  # Gemini 3.0+ defaults to 1.0
    max_tokens=500
)

query = st.chat_input("Ask a question about the PDF document:")
prompt = query

system_prompt = (
    "You are assistant for question-answering tasks"
    "Use the following pieces of context to answer the question at the end."
    "If you don't know the answer, just say that you don't know, don't try to make up an answer." \
    "Use three sentences maximum to answer."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("user", "{input}"),
    ]
)

if query:
    question_answer_chain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(retriever,question_answer_chain)
    response = rag_chain.invoke({"input": "Explain the transformer architecture?"})

    st.write(response["answer"])