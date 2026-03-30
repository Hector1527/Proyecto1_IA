# Clasificación de Solicitudes a Mesa de Ayuda
**Naïve Bayes desde cero | Universidad Rafael Landívar — IA 2026**

## Descripción
Sistema de clasificación automática de tickets de soporte usando Naïve Bayes Multinomial implementado manualmente. Clasifica tickets en: Technical issue, Billing inquiry, Product inquiry, Refund request, Cancellation request.

## Estructura del proyecto
```
helpdesk_classifier/
├── preprocessor.py   # Limpieza, tokenización, stopwords, stemming
├── naive_bayes.py    # Modelo Naïve Bayes desde cero
├── evaluator.py      # K-Folds, métricas, matriz de confusión
├── train.py          # Script de entrenamiento
├── app.py            # API Flask
├── model.pkl         # Modelo entrenado (generado por train.py)
├── requirements.txt
└── README.md
```

## Instalación
```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Colocar el dataset en la carpeta raíz con el nombre:
#    dataset.csv
```

## Uso

### Entrenar el modelo
```bash
python train.py
```
Esto ejecuta K-Folds (K=5), muestra las métricas y guarda `model.pkl`.

### Iniciar la API
```bash
python app.py
```
La API queda disponible en `http://localhost:5000`.

### Endpoints disponibles
| Método | Ruta        | Descripción                        |
|--------|-------------|------------------------------------|
| GET    | `/health`   | Verifica que el modelo esté cargado |
| POST   | `/classify` | Clasifica un ticket de texto       |

#### Ejemplo de clasificación
```bash
curl -X POST http://localhost:5000/classify \
     -H "Content-Type: application/json" \
     -d '{"text": "I cannot log into my account and need help resetting my password"}'
```

Respuesta:
```json
{
  "category": "Technical issue",
  "probabilities": {
    "Technical issue": 0.6123,
    "Billing inquiry": 0.1234,
    ...
  },
  "tokens_used": 8
}
```

## Tecnologías
- **Python 3.11+**
- **NLTK** — solo para stopwords y stemming (no para clasificación)
- **Pandas** — carga del dataset
- **Flask** — API REST
- **Sin scikit-learn ni equivalentes** — Naïve Bayes implementado manualmente

## Algoritmo
El modelo implementa **Naïve Bayes Multinomial** con:
- **Bag of Words**: representación de documentos como vectores de frecuencia
- **Laplace Smoothing**: `P(w|c) = (count(w,c) + 1) / (total(c) + |V|)` para evitar probabilidad cero
- **Suma de logaritmos**: `log P(c|d) = log P(c) + Σ log P(w|c)` para evitar underflow numérico
- **K-Folds Cross Validation** (K=5) implementado manualmente
