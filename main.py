import random
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from sentence_transformers import SentenceTransformer, util
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()  # 👈 primero defines la app

# luego montas tu carpeta estática
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# --- Conexión a MongoDB ---
try:
    uri = "mongodb+srv://afriverat24_db_user:tUjsCtHygG38wdVN@cluster0.i58fpb8.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    client = MongoClient(uri, server_api=ServerApi('1'))
    client.admin.command('ping')
    print("✅ Conectado a MongoDB Atlas.")
except Exception as e:
    print("❌ Error de conexión:", e)
    exit()

db = client["chatbot_db"]
coleccion = db["intenciones"]

def cargar_intenciones_desde_mongo():
    intenciones_dict = {}
    respuestas_dict = {}
    for doc in coleccion.find():
        intencion = doc["intencion"]
        intenciones_dict[intencion] = doc["patrones"]
        respuestas_dict[intencion] = doc["respuestas"]
    return intenciones_dict, respuestas_dict

intenciones, respuestas = cargar_intenciones_desde_mongo()
modelo = SentenceTransformer("/home/ec2-user/ChatBot_EntreHilosCrochet/models/all-MiniLM-L6-v2")
intenciones_embeddings = {intent: modelo.encode(frases, convert_to_tensor=True) for intent, frases in intenciones.items()}

def detectar_intencion(texto_usuario, umbral=0.6):
    emb_usuario = modelo.encode(texto_usuario, convert_to_tensor=True)
    mejor_intencion, mejor_score = None, -1
    for intent, embs in intenciones_embeddings.items():
        score = util.cos_sim(emb_usuario, embs).max().item()
        if score > mejor_score:
            mejor_score, mejor_intencion = score, intent
    if mejor_score < umbral:
        return None, mejor_score
    return mejor_intencion, mejor_score

# --- FastAPI ---

class MensajeUsuario(BaseModel):
    mensaje: str

@app.post("/chat")
async def chat_con_usuario(data: MensajeUsuario):
    mensaje_usuario = data.mensaje
    intencion, score = detectar_intencion(mensaje_usuario)

    if intencion is None:
        return {"respuesta": respuestas.get("desconocido", ["Perdona, aún estoy aprendiendo y no entendí bien. 😊"])[0]}
    else:
        respuesta_seleccionada = random.choice(respuestas.get(intencion, ["Perdona, aún estoy aprendiendo y no entendí bien. 😊"]))
        return {"respuesta": respuesta_seleccionada}
