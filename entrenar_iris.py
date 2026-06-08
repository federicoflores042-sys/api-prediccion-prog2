"""
Modelo 1 - Clasificación: Iris Dataset
Entrena un RandomForestClassifier y lo guarda en models/modelo_iris.pkl
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def entrenar_modelo():
    # Cargar dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Dividir en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar modelo
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Evaluar
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy del modelo Iris: {accuracy:.4f}")

    # Guardar modelo y metadata
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, "models/modelo_iris.pkl")
    joblib.dump(iris.target_names.tolist(), "models/iris_clases.pkl")
    joblib.dump(iris.feature_names, "models/iris_features.pkl")

    print("Modelo Iris guardado en models/modelo_iris.pkl")
    print(f"Clases: {iris.target_names.tolist()}")
    print(f"Features: {list(iris.feature_names)}")

if __name__ == "__main__":
    entrenar_modelo()
