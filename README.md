Asistente Conversacional Inteligente para Noticias Financieras (RAG)
Este proyecto es el resultado del Curso de Desarrollo de Asistentes Conversacionales, con el objetivo de construir un asistente conversacional que aproveche las ventajas de la Generación Aumentada por Recuperación (RAG) y tecnologías modernas como bases de datos vectoriales y frameworks especializados.

🚀 Objetivo del Proyecto
El asistente está diseñado para:
Responder preguntas de manera precisa y eficiente basándose en una base de conocimientos de una web.
Ofrecer una experiencia de usuario enriquecida a través de una interfaz gráfica amigable.
Visualizar las cotizaciones del día de algunas acciones financieras directamente en la interfaz.
Nota: Aunque estaba planificada la integración de un sistema de memoria conversacional, esta funcionalidad no fue implementada en esta versión.

🛠️ Componentes Técnicos
Generación Aumentada por Recuperación (RAG)
El asistente utiliza una base de datos vectorial para manejar su fuente de conocimiento, lo que permite responder preguntas que no están directamente cubiertas por los datos de entrenamiento del modelo base.

Base de datos vectorial:
FAISS
Framework
La solución está construida utilizando:

Langchain: Orquestación y gestión de las interacciones del asistente.
Embeddings
Se utiliza el modelo de embeddings para convertir el texto en representaciones vectoriales:

HuggingFaceEmbeddings: sentence-transformers/all-MiniLM-L6-v2
Modelo de Lenguaje
El asistente utiliza el siguiente modelo para la generación de respuestas:

ChatGroq: llama-3.2-90b-vision-preview
Interfaz de Usuario
La interfaz está diseñada para ser accesible y fácil de usar mediante Streamlit, mostrando:

Respuestas a las consultas del usuario.
Cotizaciones del día de acciones financieras seleccionadas en el lado izquierdo de la pantalla.

 Modelo
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

