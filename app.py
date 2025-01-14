from flask import Flask, request, jsonify
import os
import logging
import sys
import re
import json
import tiktoken
from datetime import datetime
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from typing import List, Dict
from langchain.callbacks import get_openai_callback

# Configuración de logging
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0125")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def extract_metadata(self, chunk: str) -> Dict:
        metadata = {
            'pagina': None,
            'capitulo': None,
            'articulo': None,
            'tema': None
        }
        lines = chunk.split('\n')
        for line in lines:
            if 'METADATA:' in line:
                if 'Página:' in line:
                    match = re.search(r'Página: (\d+)', line)
                    if match:
                        metadata['pagina'] = match.group(1)
                if 'Capítulo:' in line:
                    match = re.search(r'Capítulo: ([^,]+)', line)
                    if match:
                        metadata['capitulo'] = match.group(1).strip()
                if 'Artículo:' in line:
                    match = re.search(r'Artículo: ([^,]+)', line)
                    if match:
                        metadata['articulo'] = match.group(1).strip()
                if 'Tema:' in line:
                    match = re.search(r'Tema: (.+)', line)
                    if match:
                        metadata['tema'] = match.group(1).strip()
        return metadata


class QASystem:
    def __init__(self, chunks_file_path: str):
        self.processor = DocumentProcessor()
        self.knowledge_base = None
        self.chunks = []
        self.initialize_system(chunks_file_path)

    def initialize_system(self, chunks_file_path: str):
        try:
            logger.info("Iniciando procesamiento del archivo de texto")
            with open(chunks_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            chunks = content.split("=====")
            self.chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.knowledge_base = FAISS.from_texts(self.chunks, embeddings)
            logger.info(f"Sistema inicializado con {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error inicializando sistema: {str(e)}")

    def process_question(self, question: str, api_key: str) -> Dict:
        try:
            docs = self.knowledge_base.similarity_search(question, k=8)
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt_template = """
            ESTRUCTURA DE RESPUESTA:
            1. CITA TEXTUAL:
            - Proporciona la cita textual completa del artículo o sección relevante.
            - Incluye TODOS los metadatos disponibles (Página, Capítulo, Artículo, Tema).
            2. CONTEXTO LEGAL:
            - Menciona la ubicación exacta dentro del documento (capítulo, sección, número de página).
            - Indica si es parte de alguna reforma (si se menciona en los metadatos).
            3. ANÁLISIS:
            - Explica el contenido citado.
            - Menciona referencias cruzadas a otros artículos (si aparecen en los chunks proporcionados).
            - Proporciona ejemplos prácticos o interpretaciones legales cuando sea aplicable.
            Pregunta: {question}
            Contexto: {context}
            Respuesta detallada:
            """
            
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            llm = ChatOpenAI(
                temperature=0.3,
                model_name="ft:gpt-3.5-turbo-0125:personal:iva-finetuned-pdf-128:Aph45p7l",
                openai_api_key=api_key
            )
            
            chain = load_qa_chain(
                llm,
                chain_type="stuff",
                prompt=PROMPT
            )
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=question)
                context_used = [
                    {
                        "content": doc.page_content,
                        "metadata": self.processor.extract_metadata(doc.page_content),
                        "tokens": self.processor.count_tokens(doc.page_content)
                    } for doc in docs
                ]
                return {
                    "respuesta": response,
                    "metricas": {
                        "tokens_totales": cb.total_tokens,
                        "costo_estimado": cb.total_cost,
                        "chunks_relevantes": len(docs)
                    },
                    "contexto_usado": context_used
                }
        except Exception as e:
            logger.error(f"Error procesando pregunta: {str(e)}")
            return {"error": str(e), "tipo": "error"}


app = Flask(__name__)
qa_system = None

def init_qa_system():
    global qa_system
    qa_system = QASystem('data/chunks_d.txt')

@app.before_request
def ensure_qa_system():
    global qa_system
    if qa_system is None:
        init_qa_system()

@app.route('/consulta', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        question = data.get("pregunta")
        api_key = data.get("api_key")

        if not question or not api_key:
            return jsonify({"error": "Faltan datos requeridos."}), 400

        response = qa_system.process_question(question, api_key)
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error en /consulta: {str(e)}")
        return jsonify({"error": "Error interno"}), 500

@app.route('/status', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"}), 200

if __name__ == "__main__":
    init_qa_system()
    app.run(host="0.0.0.0", port=8080)
