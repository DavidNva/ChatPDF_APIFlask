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
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from typing import List, Dict, Optional
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
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
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

class ConversationManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True
        )
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=api_key
        )
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def detect_intent(self, text: str) -> Dict[str, any]:
        intent_prompt = f"""
        Analiza el siguiente mensaje y determina:
        1. La intención principal del usuario (saludo, despedida, pregunta legal, consulta general, etc)
        2. Si hace referencia a conversación previa
        3. El tema principal si es una consulta
        
        Mensaje: {text}
        
        Responde en formato JSON con las siguientes keys:
        - intent_type: tipo de intención
        - references_history: true/false
        - main_topic: tema principal o null
        """
        
        response = self.llm.predict(intent_prompt)
        try:
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error parsing intent response: {str(e)}")
            return {
                "intent_type": "unknown",
                "references_history": False,
                "main_topic": None
            }

    def get_conversational_response(self, text: str, intent: Dict[str, any]) -> Optional[Dict]:
        if intent["intent_type"] in ["saludo", "despedida", "agradecimiento"]:
            response_prompt = f"""
            Genera una respuesta profesional para un asistente especializado.
            Mensaje del usuario: {text}
            Tipo de intención: {intent["intent_type"]}
            """
            
            response = self.llm.predict(response_prompt)
            return {
                "respuesta": response,
                "tipo": "conversacional",
                "intent": intent
            }
        return None

    def add_to_history(self, user_message: str, assistant_response: str):
        self.memory.save_context(
            {"input": user_message},
            {"output": assistant_response}
        )

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
            conversation_manager = ConversationManager(api_key)
            intent = conversation_manager.detect_intent(question)
            conv_response = conversation_manager.get_conversational_response(question, intent)
            
            if conv_response:
                return conv_response

            docs = self.knowledge_base.similarity_search(question, k=8)
            context = "\n".join([doc.page_content for doc in docs])
            
            prompt = f"""
            Pregunta: {question}
            Contexto: {context}
            Genera una respuesta basada en el contexto proporcionado.
            """
            
            response = conversation_manager.llm.predict(prompt)
            return {"respuesta": response, "tipo": "legal"}
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