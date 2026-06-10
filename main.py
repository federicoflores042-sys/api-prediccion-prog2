"""
API REST - Trabajo Final Integrador - Programación II
Integrantes: Federico Flores, Elias Salgueiro, Valentina Gloriani
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

# ──────────────────────────────────────────────
# Inicialización de la app
# ──────────────────────────────────────────────

app = FastAPI(
    title="API de Predicción con Machine Learning",
    description=(
        "API REST desarrollada con FastAPI que permite realizar predicciones "
        "de clasificación (Iris) y regresión (Consumo de Combustible) "
        "utilizando modelos entrenados con Scikit-Learn.\n\n"
        "**Integrantes:** Federico Flores, Elias Salgueiro, Valentina Gloriani"
    ),
    version="1.0.0",
)

# ──────────────────────────────────────────────
# Carga de modelos al iniciar
# ──────────────────────────────────────────────

BASE = os.path.dirname(__file__)

try:
    modelo_iris    = joblib.load(os.path.join(BASE, "models/modelo_iris.pkl"))
    iris_clases    = joblib.load(os.path.join(BASE, "models/iris_clases.pkl"))
    iris_features  = joblib.load(os.path.join(BASE, "models/iris_features.pkl"))
except FileNotFoundError:
    raise RuntimeError(
        "No se encontraron los modelos. "
        "Ejecutá primero scripts/entrenar_iris.py y scripts/entrenar_mpg.py"
    )

try:
    modelo_mpg   = joblib.load(os.path.join(BASE, "models/modelo_mpg.pkl"))
    mpg_features = joblib.load(os.path.join(BASE, "models/mpg_features.pkl"))
    mpg_rangos   = joblib.load(os.path.join(BASE, "models/mpg_rangos.pkl"))
except FileNotFoundError:
    raise RuntimeError("No se encontró el modelo MPG.")

# ──────────────────────────────────────────────
# Schemas de entrada
# ──────────────────────────────────────────────

class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Longitud del sépalo (cm)", example=5.1)
    sepal_width:  float = Field(..., gt=0, description="Ancho del sépalo (cm)",    example=3.5)
    petal_length: float = Field(..., gt=0, description="Longitud del pétalo (cm)", example=1.4)
    petal_width:  float = Field(..., gt=0, description="Ancho del pétalo (cm)",    example=0.2)

class MPGInput(BaseModel):
    cylinders:    int   = Field(..., ge=3, le=8,   description="Número de cilindros",             example=4)
    displacement: float = Field(..., gt=0,          description="Cilindrada (pulgadas³)",          example=140.0)
    horsepower:   float = Field(..., gt=0,          description="Potencia (HP)",                   example=90.0)
    weight:       float = Field(..., gt=0,          description="Peso del vehículo (libras)",      example=2264.0)
    acceleration: float = Field(..., gt=0,          description="Aceleración (0-60 mph en seg)",  example=15.5)
    model_year:   int   = Field(..., ge=70, le=82,  description="Año del modelo (70-82)",          example=75)
    origin:       int   = Field(..., ge=1, le=3,    description="Origen: 1=USA, 2=Europa, 3=Asia", example=1)

# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/status", tags=["General"])
def status():
    """Verifica que la API está funcionando correctamente."""
    return {
        "status": "ok",
        "message": "La API está en línea y funcionando correctamente."
    }


@app.get("/info", tags=["General"])
def info():
    """Información general del proyecto, integrantes y modelos."""
    return {
        "materia": "Programación II",
        "proyecto": "API de Predicción con Machine Learning",
        "descripcion": (
            "API REST desarrollada con FastAPI que permite realizar predicciones "
            "de clasificación y regresión utilizando modelos entrenados con Scikit-Learn."
        ),
        "integrantes": [
    {"nombre": "Federico Flores",    "dni": "39.249.475", "email": "federicoflores042@gmail.com", "rol": "Desarrollador"},
    {"nombre": "Elias Salgueiro",    "dni": "37.034.994", "email": "salgueiroelias@gmail.com",   "rol": "Desarrollador"},
    {"nombre": "Valentina Gloriani", "dni": "42.053.998", "email": "valeegloriani@gmail.com",    "rol": "Desarrolladora"},
        ],
        
        "modelo_clasificacion": {
            "nombre": "Iris Species Classifier",
            "tipo": "Clasificación",
            "algoritmo": "Random Forest",
            "dataset": "Iris",
            "descripcion": "Predice la especie de una flor iris según medidas de sépalos y pétalos.",
            "clases": iris_clases,
            "endpoint": "/modelo1",
        },
        "modelo_regresion": {
            "nombre": "Fuel Consumption Predictor",
            "tipo": "Regresión",
            "algoritmo": "Gradient Boosting",
            "dataset": "Auto MPG",
            "descripcion": "Predice el consumo de combustible de un vehículo en millas por galón (mpg).",
            "endpoint": "/modelo2",
        },
        "uso_endpoints": {
            "/modelo1": {
                "metodo": "POST",
                "descripcion": "Predice la especie de iris.",
                "ejemplo_body": {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2,
                },
            },
            "/modelo2": {
                "metodo": "POST",
                "descripcion": "Predice el consumo de combustible en MPG.",
                "ejemplo_body": {
                    "cylinders": 4,
                    "displacement": 140.0,
                    "horsepower": 90.0,
                    "weight": 2264.0,
                    "acceleration": 15.5,
                    "model_year": 75,
                    "origin": 1,
                },
            },
        },
    }


@app.post("/modelo1", tags=["Predicciones"])
def predecir_iris(datos: IrisInput):
    """
    Predice la especie de una flor iris.

    - **sepal_length**: longitud del sépalo en cm
    - **sepal_width**: ancho del sépalo en cm
    - **petal_length**: longitud del pétalo en cm
    - **petal_width**: ancho del pétalo en cm
    """
    try:
        X = np.array([[
            datos.sepal_length,
            datos.sepal_width,
            datos.petal_length,
            datos.petal_width,
        ]])
        prediccion   = modelo_iris.predict(X)[0]
        probabilidades = modelo_iris.predict_proba(X)[0]
        especie      = iris_clases[prediccion]

        return {
            "prediccion": int(prediccion),
            "especie": especie,
            "probabilidades": {
                iris_clases[i]: round(float(p), 4)
                for i, p in enumerate(probabilidades)
            },
            "datos_recibidos": datos.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


@app.post("/modelo2", tags=["Predicciones"])
def predecir_mpg(datos: MPGInput):
    """
    Predice el consumo de combustible en millas por galón (MPG).

    - **cylinders**: número de cilindros (3-8)
    - **displacement**: cilindrada en pulgadas cúbicas
    - **horsepower**: potencia en HP
    - **weight**: peso del vehículo en libras
    - **acceleration**: tiempo de 0 a 60 mph en segundos
    - **model_year**: año del modelo (70 = 1970, 82 = 1982)
    - **origin**: 1 = USA, 2 = Europa, 3 = Asia
    """
    try:
        X = np.array([[
            datos.cylinders,
            datos.displacement,
            datos.horsepower,
            datos.weight,
            datos.acceleration,
            datos.model_year,
            datos.origin,
        ]])
        mpg_predicho = modelo_mpg.predict(X)[0]

        return {
            "mpg_predicho": round(float(mpg_predicho), 2),
            "litros_por_100km": round(235.21 / float(mpg_predicho), 2),
            "interpretacion": _interpretar_consumo(float(mpg_predicho)),
            "datos_recibidos": datos.model_dump(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {str(e)}")


def _interpretar_consumo(mpg: float) -> str:
    if mpg >= 35:
        return "Consumo muy eficiente"
    elif mpg >= 25:
        return "Consumo eficiente"
    elif mpg >= 18:
        return "Consumo moderado"
    else:
        return "Consumo alto"
