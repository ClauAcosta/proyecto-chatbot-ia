# langchain_setup.py

from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

# Cargar documentos desde las URLs
def load_documents():
    page_urls = [
        "https://www.bloomberglinea.com/latinoamerica/argentina/",
        ]
    docs = []
    for page_url in page_urls:
        try:
            loader = WebBaseLoader(web_path=[page_url])
            docs_lazy = loader.lazy_load()
            for doc in docs_lazy:
                docs.append(doc)
        except Exception as e:
            print(f"Error al cargar contenido de {page_url}: {e}")
    return docs

# Configurar LangChain
def setup_langchain(docs):
    # Dividir documentos en fragmentos
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n", "\n\n", "."]
    )
    documents = splitter.split_documents(docs)

    # Configurar el modelo y el vectorstore
    llm = ChatGroq(model="llama-3.2-90b-vision-preview")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    # Configurar la memoria de conversación
    memory = ConversationBufferMemory()

    # Configurar la plantilla del prompt
    template = """Responder las preguntas basado en el siguiente contexto:
    {context}

    Pregunta: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    # Crear la cadena de procesamiento
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    
    return chain, memory

