from flask import Flask, request, jsonify
import os
import logging
import sys
import re
import tiktoken
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from typing import List, Dict
from langchain.callbacks import get_openai_callback

# Configuración de logging para Cloud Run
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentProcessor:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

class QASystem:
    def __init__(self, chunks_file_path: str):
        self.processor = DocumentProcessor()
        self.knowledge_base = None
        self.chunks = None
        self.initialize_system(chunks_file_path)

    def initialize_system(self, chunks_file_path: str):
        try:
            logging.info("Iniciando procesamiento del archivo de texto")
            
            # Usar ruta absoluta para el archivo de chunks
            absolute_chunks_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), chunks_file_path)
            
            # Leer el archivo de chunks
            with open(absolute_chunks_path, 'r', encoding='utf-8') as file:
                text_content = file.read()
            
            # Procesar chunks
            chunks = []
            current_chunk = ""
            
            for line in text_content.split('\n'):
                if line.strip() == "==================================================":
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = ""
                else:
                    current_chunk += line + '\n'
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            self.chunks = [chunk for chunk in chunks if chunk]
            logging.info(f"Documento dividido en {len(self.chunks)} chunks")
            
            # Crear embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            
            self.knowledge_base = FAISS.from_texts(self.chunks, embeddings)
            logging.info("Base de conocimiento creada exitosamente")
            
        except Exception as e:
            logging.error(f"Error en initialize_system: {str(e)}")
            raise

    def process_question(self, question: str, api_key: str):
        try:
            logging.info(f"Procesando pregunta: {question}")
            
            # Configurar OpenAI API key
            os.environ["OPENAI_API_KEY"] = api_key
            
            # Búsqueda de documentos relevantes
            article_match = re.search(r'art[ií]culo\s+(\d+)', question.lower())
            k_docs = 8 if article_match else 5
            
            docs = self.knowledge_base.similarity_search(question, k=k_docs)
            
            if article_match:
                article_num = article_match.group(1)
                article_pattern = re.compile(f'art[ií]culo\s+{article_num}\\b', re.IGNORECASE)
                docs = sorted(docs, key=lambda x: 1 if article_pattern.search(x.page_content) else 0, reverse=True)
            
            # Filtrar documentos por tokens
            total_tokens = 0
            filtered_docs = []
            
            for doc in docs:
                tokens = self.processor.count_tokens(doc.page_content)
                if total_tokens + tokens < 14000:
                    total_tokens += tokens
                    filtered_docs.append(doc)
            
            # Configurar el prompt y el modelo
            prompt_template = """
            Eres un asistente especializado en análisis de la Ley del Impuesto al Valor Agregado y otros documentos legales con instrucciones específicas de proporcionar respuestas detalladas y exhaustivas.

            INSTRUCCIONES PRINCIPALES: 
            1. SIEMPRE responde basándote ÚNICAMENTE en la información proporcionada en los chunks del documento. 
            2. Cita TEXTUALMENTE las partes relevantes del documento, incluyendo TODOS los metadatos disponibles (página, capítulo, artículo). 
            3. Si la información solicitada no está en los chunks proporcionados, indica claramente: "No encuentro información específica sobre esto en los fragmentos proporcionados del documento."

            ### Cómo manejar los metadatos:
            # - Siempre menciona la **página**, **capítulo**, y **artículo** en la respuesta, tal como aparece en los metadatos.
            # - Si el artículo tiene incisos o partes relacionadas (como "5o.- Inciso VI, derivados a)"), indícalos explícitamente en tu análisis. - Utiliza el campo "Tema" en los metadatos para contextualizar y enfocar la respuesta.

            ESTRUCTURA DE RESPUESTA: 
            1. CITA TEXTUAL:
            - Primero, proporciona la cita textual completa del artículo o sección relevante
            - Incluye TODOS los metadatos disponibles (Página, Capítulo, Artículo, Tema)
            - Si solicitan mas detalle, agrega informacion sobre los incisos o puntos del articulos, ten en cuenta chunks relacionados de los articulos.
            2. CONTEXTO LEGAL Documento:
            - Menciona la ubicación exacta dentro del documento (capítulo, sección, numero de pagin)
            - Indica si es parte de alguna reforma (si se menciona en los metadatos)
            
            3. ANÁLISIS (solo si toda la información anterior está disponible):
            - Explica el contenido citado
            - Menciona cualquier referencia cruzada a otros artículos (solo si aparecen en los chunks proporcionados)
            - Proporciona ejemplos prácticos o interpretaciones legales cuando sea aplicable.

            2. Para preguntas generales:
               - Incluye TODAS las menciones relevantes del tema
               - Busca en los *chunks* correspondientes y **recopila todas las menciones relevantes al tema consultado**.
            3. Para cualquier respuesta:
               - **No resumas** ni omitas información relevante. Sé claro y preciso en el lenguaje.
               - Usa citas textuales extensas cuando estén disponibles en los metadatos o contenido del documento.
               - Mantén la precisión y un lenguaje legal adecuado.
               - Indica si hay información o artículos relacionados adicionales.

            Contexto: {context}
            
            Pregunta: {question}
            
            Respuesta detallada:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            llm = ChatOpenAI(
                model_name='gpt-3.5-turbo',
                temperature=0.2,
                max_tokens=2000
            )
            
            chain = load_qa_chain(
                llm, 
                chain_type="stuff",
                prompt=PROMPT
            )
            
            # Generar respuesta y obtener métricas
            with get_openai_callback() as cb:
                respuesta = chain.run(input_documents=filtered_docs, question=question)
                
                # Preparar contexto usado
                context_used = [
                    {
                        "content": doc.page_content,
                        "tokens": self.processor.count_tokens(doc.page_content)
                    }
                    for doc in filtered_docs
                ]
                
                return {
                    "respuesta": respuesta,
                    "metricas": {
                        "tokens_totales": cb.total_tokens,
                        "costo_estimado": cb.total_cost,
                        "chunks_relevantes": len(filtered_docs)
                    },
                    "contexto_usado": context_used
                }
                
        except Exception as e:
            logging.error(f"Error en process_question: {str(e)}")
            raise

# Inicializar Flask app
app = Flask(__name__)

# Ruta al archivo de chunks relativa al directorio de la aplicación
CHUNKS_FILE_PATH = os.path.join('data', 'chunks_d.txt')

# Variable global para el sistema QA
qa_system = None

def init_qa_system():
    global qa_system
    if qa_system is None:
        try:
            qa_system = QASystem(CHUNKS_FILE_PATH)
            logging.info("Sistema QA inicializado correctamente")
        except Exception as e:
            logging.error(f"Error inicializando QA system: {str(e)}")
            raise

# @app.before_first_request
# def initialize():
#     init_qa_system()
@app.before_request
def initialize_qa_system():
    global qa_system
    if qa_system is None:
        logging.info("Inicializando sistema QA antes de la primera solicitud")
        init_qa_system()

@app.route('/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            "status": "OK",
            "message": "API is running",
            "version": "1.0"
        })
    except Exception as e:
        logging.error(f"Error en health check: {str(e)}")
        return jsonify({
            "status": "ERROR",
            "message": str(e)
        }), 500

@app.route('/consulta', methods=['POST'])
def process_query():
    try:
        if qa_system is None:
            init_qa_system()
        
        data = request.json
        if not data or 'pregunta' not in data or 'api_key' not in data:
            return jsonify({
                "error": "Se requiere 'pregunta' y 'api_key' en el cuerpo de la solicitud"
            }), 400
        
        result = qa_system.process_question(data['pregunta'], data['api_key'])
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Error en endpoint /consulta: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)