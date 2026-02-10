from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import requests
import psycopg2


# Inicialización FastAPI

app = FastAPI(title="TMDB API")

# modelo y preprocesadores

try:
    with open("movie_model.pkl", "rb") as f:
        package = pickle.load(f)

    numeric_features = package['numeric_features']
    genre_columns = package['genre_columns']
    scaler = package['scaler']
    tfidf_vectorizer = package['tfidf_vectorizer']
    pca = package['pca']
    gb_model = package['gb_model']


except FileNotFoundError:
    print("movie_model.pkl no encontrado. El endpoint /predict no funcionará.")
    gb_model, scaler, tfidf_vectorizer, pca, numeric_features, genre_columns = [None]*6
    
# Conexión PostgreSQL 
def get_connection():
    return psycopg2.connect(
        user=os.getenv("DB_USER"), 
        password=os.getenv("DB_PASSWORD"), 
        host=os.getenv("DB_HOST"), 
        port=int(os.getenv("DB_PORT")), 
        dbname=os.getenv("DB_NAME")) 



# Modelos Pydantic

class MovieInput(BaseModel):
    title: str
    overview: str = ""
    tagline: str = ""
    budget: float
    runtime: float
    popularity: float
    vote_count: int
    release_year: int
    adult: bool
    video: bool
    genres: list[str] = []

class Question(BaseModel):
    text: str


# modelo Hugging Face 

HF_TOKEN = os.getenv("HF_TOKEN") #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  
SQL_MODEL_ID = "defog/sqlcoder-7b-2"
TEXT_MODEL_ID = "tiiuae/falcon-7b-instruct"

def query_hf_api(prompt: str, model_id: str):
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 250, "temperature": 0.1}}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        if response.status_code != 200:
            return f"Error: {response.json().get('error', 'Servicio no disponible')}"
        return response.json()[0]['generated_text']
    except Exception as e:
        return f"Error de conexión con la IA: {str(e)}"

def generar_sql(prompt: str) -> str:
    return query_hf_api(prompt, SQL_MODEL_ID)

def generar_texto(prompt: str) -> str:
    return query_hf_api(prompt, TEXT_MODEL_ID)


# SQL Schema

sql_schema = """
TABLE genres (
    genre_id INTEGER,
    name TEXT
);

TABLE movie_genres (
    movie_id BIGINT,
    genre_id INTEGER
);

TABLE movies (
    movie_id BIGINT,
    title TEXT,
    original_title TEXT,
    overview TEXT,
    release_date DATE,
    runtime INT, 
    budget BIGINT,
    revenue BIGINT, 
    popularity DOUBLE PRECISION,
    vote_average DOUBLE PRECISION,
    vote_count INTEGER,
    status TEXT,
    tagline TEXT,
    imdb_id TEXT,
    original_language TEXT,
    adult BOOLEAN,
    video BOOLEAN
);
"""


# endpoint /predict

@app.post("/predict")
def predict_movie(movie: MovieInput):
    if gb_model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")

    # features numéricas
    X_num = np.array([[movie.budget,
                       movie.runtime,
                       movie.popularity,
                       movie.vote_count,
                       movie.release_year,
                       int(movie.adult),
                       int(movie.video)]], dtype=float)

    # géneros one-hot
    X_genres = np.zeros((1, len(genre_columns)))
    movie_genres = [x.lower().strip() for x in movie.genres]
    for i, g in enumerate(genre_columns):
        if g.lower() in movie_genres:
            X_genres[0, i] = 1


    # texto
    text_combined = f"{movie.title} {movie.overview} {movie.tagline}"
    X_text_tfidf = tfidf_vectorizer.transform([text_combined]).toarray()
    X_text_pca = pca.transform(X_text_tfidf)

    # combina features
    X_final = np.hstack([X_num, X_genres, X_text_pca])
    X_final_scaled = scaler.transform(X_final)

    pred_class = int(gb_model.predict(X_final_scaled)[0])
    pred_prob = float(gb_model.predict_proba(X_final_scaled)[0, 1])

    return {
        "success_prediction": pred_class,
        "success_probability": pred_prob
    }


# endpoint /ask-text

@app.post("/ask-text")
def ask_text(question: Question) -> str:
    prompt = f"""
    Eres un bot experto en SQL. Base de datos:
    {sql_schema}
    Pregunta del usuario:
    {question.text}
    Limita resultados a 50 filas.
    """
    respuesta_sql = generar_sql(prompt).strip()
    sql_lower = respuesta_sql.lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"]

    if not sql_lower.startswith("select") or any(w in sql_lower for w in forbidden):
        raise HTTPException(400, "Consulta SQL no permitida")

    with get_connection() as conn:
        df = pd.read_sql(respuesta_sql, conn)
    if df.empty:
        return "No hay resultados para esa consulta."

    prompt_respuesta = f"""
    Eres un bot que responde de forma clara y humana.
    Pregunta: {question.text}
    Datos encontrados: {df.head(10).to_dict(orient='records')}
    """
    respuesta_final = generar_texto(prompt_respuesta).strip()
    return respuesta_final


# endpoint /ask-visual

@app.post("/ask-visual")
def ask_visual(question: Question) -> StreamingResponse:
    prompt_graficos = f"""
    Eres un bot que genera SQL para visualización. Base de datos:
    {sql_schema}
    Pregunta: {question.text}
    """
    respuesta = generar_sql(prompt_graficos).strip()
    lines = respuesta.splitlines()
    if len(lines) < 2:
        raise HTTPException(400, "Respuesta inválida del modelo")

    plot_type_raw = lines[0].strip().upper()
    plot_map = {"HISTOGRAMA": "hist", "SCATTERPLOT": "scatter", "BOXPLOT": "box"}
    if plot_type_raw not in plot_map:
        raise HTTPException(400, "Tipo de gráfico no reconocido")
    plot_type = plot_map[plot_type_raw]

    respuesta_sql = " ".join(lines[1:]).strip()
    sql_lower = respuesta_sql.lower()
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke"]

    if not sql_lower.startswith("select") or any(w in sql_lower for w in forbidden) or "limit" not in sql_lower:
        raise HTTPException(400, "Consulta SQL no permitida o falta LIMIT")
    with get_connection() as conn:
        df = pd.read_sql(respuesta_sql, conn)
    if df.empty:
        raise HTTPException(404, "No hay datos para generar el gráfico")

    fig, ax = plt.subplots(figsize=(7,5))
    if plot_type == "hist":
        ax.hist(df.iloc[:,0])
        ax.set_xlabel(df.columns[0])
        ax.set_title("Histograma")
    elif plot_type == "scatter":
        if df.shape[1] < 2:
            raise HTTPException(400, "Scatterplot requiere dos columnas")
        ax.scatter(df.iloc[:,0], df.iloc[:,1])
        ax.set_xlabel(df.columns[0])
        ax.set_ylabel(df.columns[1])
        ax.set_title("Scatterplot")
    elif plot_type == "box":
        ax.boxplot(df.iloc[:,0])
        ax.set_ylabel(df.columns[0])
        ax.set_title("Boxplot")

    buffer_bytes = BytesIO()
    fig.savefig(buffer_bytes, format="png")
    plt.close(fig)
    buffer_bytes.seek(0)
    return StreamingResponse(content=buffer_bytes, media_type="image/png")
