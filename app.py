import gradio as gr
import joblib
import numpy as np

model = joblib.load("random_forest_model.joblib")
scaler = joblib.load("scaler.joblib")

UMBRAL = 0.374  

def predecir_riesgo(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    
    sex_val = 1 if sex == "Hombre" else 0
    fbs_val = 1 if fbs == "Sí" else 0
    exang_val = 1 if exang == "Sí" else 0

    X = np.array([[age, sex_val, cp, trestbps, chol, fbs_val, restecg, thalach, exang_val, oldpeak, slope, ca, thal]])
    
    X_scaled = scaler.transform(X)
    prob = model.predict_proba(X_scaled)[0][1]
    
    if prob >= UMBRAL:
        resultado = "RIESGO DE ENFERMEDAD CARDÍACA"
        color = "🔴"
    else:
        resultado = "SIN RIESGO DETECTADO"
        color = "🟢"
    
    return f"""
    {color} **{resultado}**
    
    **Probabilidad de riesgo:** {prob:.1%}
    **Umbral usado:** {UMBRAL} (optimizado para máxima sensibilidad clínica)
    
    ---
    *Modelo: Random Forest | Recall en clase de riesgo: 92.9%*
    *Este resultado es orientativo y no reemplaza un diagnóstico médico profesional.*
    """

demo = gr.Interface(
    fn=predecir_riesgo,
    inputs=[
        gr.Slider(18, 100, value=50, label="Edad", step=1),
        gr.Radio(["Hombre", "Mujer"], value="Hombre", label="Sexo"),
        gr.Dropdown(choices=[("Angina típica", 0), ("Angina atípica", 1), ("Dolor no anginoso", 2), 
        ("Asintomático", 3)], value=0, label="Tipo de dolor de pecho (cp)"
        ),
        gr.Slider(80, 220, value=130, label="Presión arterial en reposo (trestbps)"),
        gr.Slider(100, 600, value=240, label="Colesterol sérico (chol)"),
        gr.Radio(["Sí", "No"], value="No", label="Glucosa en ayunas > 120 mg/dl (fbs)"),
        gr.Dropdown(
            choices=[("Normal", 0), ("Anomalía onda ST-T", 1), ("Hipertrofia VI", 2)],
            value=0, label="ECG en reposo (restecg)"
        ),
        gr.Slider(60, 220, value=150, label="Frecuencia cardíaca máxima (thalach)"),
        gr.Radio(["Sí", "No"], value="No", label="Angina inducida por ejercicio (exang)"),
        gr.Slider(0, 7, value=1.0, label="Depresión ST (oldpeak)", step=0.1),
        gr.Dropdown(
            choices=[("Ascendente", 0), ("Plana", 1), ("Descendente", 2)],
            value=0, label="Pendiente del segmento ST (slope)"
        ),
        gr.Slider(0, 3, value=0, label="Vasos coloreados por fluoroscopía (ca)", step=1),
        gr.Dropdown(
            choices=[("Normal", 0), ("Defecto fijo", 1), ("Defecto reversible", 2)],
            value=0, label="Talasemia (thal)"
        ),
    ],
    outputs=gr.Markdown(label="Resultado"),
    title="Sistema Inteligente de Evaluación de Riesgo Cardíaco",
    description="""
    Proyecto académico — Aprendizaje Estadístico, UPAO  
    Modelo: Random Forest entrenado sobre el Cleveland Heart Disease Dataset (UCI)  
    
    Ingresa los valores clínicos del paciente para obtener una evaluación de riesgo.
    """,
    theme=gr.themes.Soft(),
)

demo.launch()