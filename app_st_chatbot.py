# app.py
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la aplicación Streamlit
st.title("📰 Asistente de Noticias Financieras")
st.markdown("### ¡Bienvenido! Hazme cualquier pregunta sobre las últimas noticias financieras y veré cómo puedo ayudarte.")


# Cargar documentos desde las URLs
page_urls = [
    "https://www.bloomberglinea.com/latinoamerica/argentina/",
]

docs = []
for page_url in page_urls:
    try:
        loader = WebBaseLoader(web_path=[page_url])
        docs_lazy = loader.lazy_load()
        docs.extend(docs_lazy)
    except Exception as e:
        st.error(f"Error al cargar contenido de {page_url}: {e}")

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

# Interfaz de usuario para preguntas
user_input = st.text_input("Haz tu pregunta sobre finanzas:")

if user_input:
    # Invocar la cadena con la memoria
    response = chain.invoke(user_input)
    
    # Guardar el contexto en la memoria
    memory.save_context({"question": user_input}, {"response": response})
    
    # Mostrar la respuesta
    st.write("Respuesta:", response)

# Mostrar el contenido de la memoria
#if st.button("Mostrar historial de conversación"):
 #   st.write(memory.buffer)
 # Sección para mostrar cotizaciones
st.sidebar.header("📈Cotizaciones del Día")
tickers = ["BTC-USD", "SPY", "QQQ", "NVDA", "YPF", "GGAL","PAM","MSTR","COIN"]
data = yf.download(tickers, period="1d", interval="1d")

# Crear un DataFrame solo con los precios de cierre
if not data.empty:
    quotes_df = data['Close'].iloc[-1]  # Obtener el último precio de cierre
    quotes_df = quotes_df.reset_index()  # Reiniciar el índice para convertirlo en DataFrame
    quotes_df.columns = ['Ticker', 'Cotización']  # Renombrar columnas
else:
    quotes_df = pd.DataFrame(columns=["Ticker", "Cotización"])

# Mostrar las cotizaciones en un dataframe
st.sidebar.subheader("Cotizaciones")
st.sidebar.dataframe(quotes_df)

