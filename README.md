Asistente Conversacional Inteligente (RAG)

Este proyecto es el resultado del Curso de Desarrollo de Asistentes Conversacionales. El objetivo es construir un asistente que responda preguntas de manera precisa y eficiente utilizando Generación Aumentada por Recuperación (RAG), integrando tecnologías modernas para bases de datos vectoriales, memoria, y frameworks como Langchain o Langgraph.
🚀 Objetivo del Proyecto
Desarrollo de un asistente conversacional que pueda:
•	Responder preguntas sobre una base de conocimientos específica.
•	Mantener una conversación fluida gracias a su sistema de memoria.
🛠️ Componentes Técnicos
1. Generación Aumentada por Recuperación (RAG)
El asistente utiliza una base de datos vectorial para manejar su fuente de conocimiento. Esto permite responder preguntas que no están directamente cubiertas por los datos de entrenamiento del modelo base.
Bases de datos vectoriales :
•	FAISS
2. Framework
La solución se construye utilizando:
•	Langchain  para la implementación y orquestación del asistente.
3. Memoria
•	Memoria conversacional para continuar interacciones de manera coherente.
4. Interfaz de Usuario
El asistente está diseñado para ser utilizado a través de:
•	Streamlit (implementación principal)

5. Modelo
 llm = ChatGroq(model="llama-3.2-90b-vision-preview")
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

📋 Requisitos de Instalación
1. Clonar el Repositorio
git clone https://github.com/ClauAcosta/proyecto-chatbot-ia.git
2. Instalar Dependencias
pip install -r requirements.txt
3. Configuración
•	Proporciona las credenciales necesarias para acceso a la base de datos vectorial y API de internet.
•	Configura las variables de entorno en un archivo .env.
🖥️ Uso
1.	Inicia la aplicación:
streamlit run app_st_chatbot.py
2.	Accede a la interfaz en http://localhost:8501.
3.	Realiza preguntas o explora las funcionalidades del asistente.

