from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Definir la estructura de entrada
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI(title="Iris API")

# Ruta absoluta para evitar errores de "File Not Found"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_iris.pkl')

# Cargar el modelo
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error: No se encontró el archivo del modelo en {MODEL_PATH}")

@app.post("/predict")
def predict(iris: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado en el servidor.")
    
    try:
        # Los modelos entrenados con DataFrame esperan los nombres de columnas originales
        # Ajusta estos nombres si tu CSV 'iris.csv' usaba otros diferentes
        df = pd.DataFrame([iris.dict().values()], 
                          columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        
        prediction = model.predict(df)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))