from flask import Flask, request, jsonify, session
import os
import logging
import sys
import re
import json
import tiktoken
from datetime import datetime, timedelta
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Optional, Tuple
from langchain.callbacks import get_openai_callback
from flask_session import Session

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
            'articulo': [],
            'tema': None,
            'es_continuacion': False
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
                    matches = re.findall(r'[\w\.-]+', line[line.find('Artículo:'):])
                    metadata['articulo'].extend([m.strip() for m in matches if m.strip()])
                if 'Tema:' in line:
                    match = re.search(r'Tema: (.+)', line)
                    if match:
                        metadata['tema'] = match.group(1).strip()

        metadata['es_continuacion'] = any('continuacion' in line.lower() for line in lines)
        return metadata

class ConversationManager:
    def __init__(self, api_key: str, memory: ConversationBufferMemory):
        self.api_key = api_key
        self.memory = memory
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="ft:gpt-3.5-turbo-0125:personal:iva-finetuned-pdf-128:Aph45p7l",
            openai_api_key=api_key
        )

    def detect_intent(self, text: str) -> Dict:
        intent_prompt = """
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

        response = self.llm.predict(intent_prompt.format(text=text))
        try:
            return json.loads(response)
        except:
            return {
                "intent_type": "unknown",
                "references_history": False,
                "main_topic": None
            }

    def get_conversational_response(self, text: str, intent: Dict) -> Optional[str]:
        responses = {
            "saludo": [
                "¡Hola! Soy un asistente especializado en la LEY DEL IMPUESTO AL VALOR AGREGADO. ¿En qué puedo ayudarte?",
                "¡Buen día! Estoy aquí para resolver tus dudas sobre la LEY DEL IMPUESTO AL VALOR AGREGADO",
            ],
            "despedida": [
                "¡Hasta luego! No dudes en volver si tienes más preguntas sobre la LEY DEL IMPUESTO AL VALOR AGREGADO.",
                "¡Que tengas un excelente día! Estoy aquí para cuando necesites más información.",
            ],
            "agradecimiento": [
                "¡De nada! ¿Hay algo más en lo que pueda ayudarte?",
                "Es un placer ayudarte. ¿Tienes alguna otra consulta sobre la LEY DEL IMPUESTO AL VALOR AGREGADO?",
            ]
        }

        if intent["intent_type"] in responses:
            from random import choice
            return choice(responses[intent["intent_type"]])
        return None

    def add_to_history(self, question: str, answer: str):
        # Aquí se guarda el historial de la conversación
        self.memory.save_context(
            {"input": question},
            {"output": answer}
        )

    def load_history(self, historial: str):
        # Carga el historial recibido por parámetro en el formato adecuado
        if historial:
            items = historial.split(" ===== ")
            for item in items:
                question, answer = item.split(" ||| ")
                self.add_to_history(question.strip(), answer.strip())

class QueryAnalyzer:
    def __init__(self):
        self.patterns = {
            'articulo': [
                r'art[íi]culo\s*[\w\.-]+',
                r'\bart\.\s*[\w\.-]+',
                r'\b\d+[A-Za-z]?[-\s]?[A-Za-z]?\b'
            ],
            'tema': [
                r'(sobre|acerca de|respecto a)\s+.+',
                r'(qué|como|cuando|donde)\s+.+\s+(en|para|sobre)\s+.+'  
            ]
        }

    def analyze_query(self, question: str) -> Dict:
        result = {
            'tipo': 'general',
            'articulos_mencionados': [],
            'temas_detectados': []
        }

        for pattern in self.patterns['articulo']:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                result['articulos_mencionados'].append(match.group())

        if result['articulos_mencionados']:
            result['tipo'] = 'articulo_especifico'

        return result

class QASystem:
    def __init__(self, chunks_file_path: str):
        self.processor = DocumentProcessor()
        self.analyzer = QueryAnalyzer()
        self.knowledge_base = None
        self.chunks = []
        self.initialize_system(chunks_file_path)

    def initialize_system(self, chunks_file_path: str):
        try:
            logger.info("Iniciando procesamiento del archivo de texto")
            with open(chunks_file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            chunks = content.split("==================================================")
            self.chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'}
            )

            texts = self.chunks
            metadatas = [self.processor.extract_metadata(chunk) for chunk in self.chunks]
            self.knowledge_base = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

            logger.info(f"Sistema inicializado con {len(self.chunks)} chunks")
        except Exception as e:
            logger.error(f"Error inicializando sistema: {str(e)}")
            raise

    def process_question(self, question: str, api_key: str, historial: Optional[str] = "") -> Dict:
        try:
            # Inicializar el ConversationManager con el historial, si existe
            conversation_manager = ConversationManager(api_key, ConversationBufferMemory(memory_key="chat_history", return_messages=True))
            
            # Cargar el historial en la memoria si se recibe uno
            if historial:
                conversation_manager.load_history(historial)

            intent = conversation_manager.detect_intent(question)

            # Manejar respuestas conversacionales
            if intent["intent_type"] in ["saludo", "despedida", "agradecimiento"]:
                response = conversation_manager.get_conversational_response(question, intent)
                
                # Crear métricas y contexto vacío para mantener consistencia
                metrics = {
                    "tokens_totales": 0,
                    "costo_estimado": 0.0,
                    "chunks_relevantes": 0
                }
                context_used = []

                return {
                    "respuesta": response,
                    "tipo": "conversacional",
                    "intent": intent,
                    "metricas": metrics,
                    "contexto_usado": context_used
                }

            # Analizar la consulta para búsqueda específica
            analysis = self.analyzer.analyze_query(question)
            k_docs = 8 if analysis['tipo'] == 'articulo_especifico' else 5
            docs = self.knowledge_base.similarity_search(question, k=k_docs)

            # Generar la respuesta completa con los tres campos
            cita_textual = "\n".join([f"- {doc.page_content.strip()}" for doc in docs])  # Concatenar citas textuales
            contexto_legal = "\n".join([f"- Capítulo: {doc.metadata['capitulo']}, Artículo: {doc.metadata['articulo']}, Página: {doc.metadata['pagina']}" for doc in docs])  # Metadatos de contexto
            analisis = "Aquí se explica el contexto legal relacionado con el artículo mencionado y los detalles pertinentes."  # Ajusta el análisis

            response = f"1. CITA TEXTUAL:\n{cita_textual}\n\n2. CONTEXTO LEGAL:\n{contexto_legal}\n\n3. ANALISIS:\n{analisis}"

            conversation_manager.add_to_history(question, response)

            return {
                "respuesta": response,
                "tipo": "legal",
                "intent": intent,
                "metricas": {
                    "tokens_totales": 0,  # Calcula los valores de tokens
                    "costo_estimado": 0.0,
                    "chunks_relevantes": len(docs)
                },
                "contexto_usado": [{"content": doc.page_content} for doc in docs]
            }

        except Exception as e:
            logger.error(f"Error procesando pregunta: {str(e)}")
            return {"error": str(e), "tipo": "error"}
# Continuación de la API (segunda mitad)

# Configuración de Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default_secret_key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)
Session(app)

# Inicialización del sistema
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
        historial = data.get("historial", "")

        if not question or not api_key:
            return jsonify({"error": "Faltan datos requeridos."}), 400

        # Procesar la pregunta y obtener la respuesta del sistema QA
        response = qa_system.process_question(question, api_key, historial)

        # Enviar la respuesta al usuario
        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error en /consulta: {str(e)}")
        return jsonify({"error": "Error interno"}), 500

@app.route('/status', methods=['GET'])
def health_check():
    return jsonify({
        "status": "OK",
        "timestamp": datetime.now().isoformat()
    }), 200

if __name__ == "__main__":
    init_qa_system()
    app.run(host="0.0.0.0", port=8080)
