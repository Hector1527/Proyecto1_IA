# Proyecto 1 IA 2026 - Help Desk Ticket Classification

Clasificador de tickets de soporte implementado con **Multinomial Naive Bayes desde cero**. El proyecto entrena el modelo con el dataset `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`, evalua su desempeno con **5-fold cross validation estratificado** y expone una **API Flask** con interfaz web para probar predicciones.

## Estado actual del proyecto

La version actual ya no usa el dataset `customer_support_tickets.csv` como fuente principal de entrenamiento. El pipeline activo trabaja con el dataset de Bitext y clasifica tickets en **11 categorias**:

- `ACCOUNT`
- `CANCEL`
- `CONTACT`
- `DELIVERY`
- `FEEDBACK`
- `INVOICE`
- `ORDER`
- `PAYMENT`
- `REFUND`
- `SHIPPING`
- `SUBSCRIPTION`

## Caracteristicas principales

- Implementacion manual de **Multinomial Naive Bayes**
- Preprocesamiento de texto con limpieza, stopwords, stemming y bigramas
- **5-fold cross validation estratificado** sin `scikit-learn`
- Persistencia del modelo entrenado en `model.pkl`
- API REST en Flask
- Interfaz web para probar tickets manualmente
- Script adicional para generar reporte y matriz de confusion sobre el modelo guardado

## Estructura del proyecto

```text
Proyecto1_IA/
|-- app.py
|-- train.py
|-- matriz.py
|-- preprocessor.py
|-- naive_bayes.py
|-- evaluator.py
|-- model.pkl
|-- Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv
|-- customer_support_tickets.csv
|-- templates/
|   `-- index.html
|-- static/
|   |-- app.js
|   `-- styles.css
|-- requirements.txt
`-- README.md
```

Notas:

- `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv` es el dataset activo del pipeline.
- `customer_support_tickets.csv` permanece en el repositorio, pero no es utilizado por `train.py` en la version actual.
- `matriz.py` evalua el `model.pkl` ya entrenado sobre el dataset completo y muestra la matriz de confusion.

## Requisitos

- Python 3.11 o superior
- Dependencias listadas en `requirements.txt`

Instalacion recomendada:

```bash
python -m venv venv
```

Activacion del entorno virtual:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Instalacion de dependencias:

```bash
pip install -r requirements.txt
```

## Dataset utilizado

Archivo activo:

- `Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`

Columnas presentes en el archivo:

- `flags`
- `instruction`
- `category`
- `intent`
- `response`

Columnas usadas por el entrenamiento:

- Texto: `instruction`
- Etiqueta: `category`

Tamano del dataset analizado en esta version:

- `26,872` registros

## Entrenamiento

Para reentrenar el modelo:

```bash
python train.py
```

El script:

1. carga el dataset activo,
2. preprocesa el texto,
3. ejecuta validacion cruzada estratificada de 5 folds,
4. imprime metricas por clase y metricas globales,
5. entrena el modelo final con todo el dataset,
6. guarda el resultado en `model.pkl`.

### Metricas de la version actual

Resultado obtenido al ejecutar `python train.py` localmente sobre la version actual del proyecto:

- Accuracy promedio: `0.9929 +- 0.0007`
- Macro F1 promedio: `0.9927 +- 0.0007`
- Vocabulario final del modelo: `10,894` tokens
- Promedio de tokens por documento preprocesado: `6.6`

Estas metricas pueden cambiar si se modifica el preprocesamiento, la configuracion del entrenamiento o el dataset.

## Evaluacion adicional

Para generar un reporte con matriz de confusion usando el modelo guardado:

```bash
python matriz.py
```

Importante: este script evalua el `model.pkl` sobre el dataset completo cargado en memoria. Sirve para inspeccion y depuracion, pero no reemplaza la validacion cruzada reportada por `train.py`.

## Ejecutar la aplicacion web

```bash
python app.py
```

La aplicacion queda disponible en:

```text
http://localhost:5000
```

## Interfaz web

La UI permite ingresar:

- `Ticket ID`
- `Channel`
- `Subject`
- `Description`

Notas sobre la interfaz:

- `Ticket ID` se autogenera desde el frontend.
- `Channel` se usa como dato visual en la UI, pero el backend actual no lo usa para clasificar.
- La prediccion se calcula a partir de `subject` y `description`.

## API disponible

### `GET /`

Devuelve la interfaz web.

### `GET /health`

Devuelve el estado del backend, el tamano del vocabulario y las clases cargadas por el modelo.

Respuesta esperada:

```json
{
  "status": "ok",
  "vocabulary_size": 10894,
  "classes": [
    "ACCOUNT",
    "CANCEL",
    "CONTACT",
    "DELIVERY",
    "FEEDBACK",
    "INVOICE",
    "ORDER",
    "PAYMENT",
    "REFUND",
    "SHIPPING",
    "SUBSCRIPTION"
  ]
}
```

### `POST /classify`

Clasifica un ticket a partir de un JSON.

El backend acepta cualquiera de estas entradas:

- `text`
- `subject` y/o `description`
- `ticket_id` opcional

Si se manda `subject` y `description`, el backend usa `preprocess_ticket()`. Si se manda `text`, usa `preprocess()`. Campos adicionales son ignorados por la API actual.

Ejemplo de solicitud:

```bash
curl -X POST http://localhost:5000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"ticket_id\":\"TCK-20260407-002\",\"subject\":\"My order still has not arrived\",\"description\":\"The tracking page has not updated in days and I need to know where my package is.\"}"
```

Ejemplo de respuesta:

```json
{
  "ticket_id": "TCK-20260407-002",
  "subject": "My order still has not arrived",
  "normalized_text": "order not arriv track page not updat day need packag order__not not__arriv arriv__track track__page page__not not__updat updat__day day__need need__packag",
  "category": "DELIVERY",
  "probabilities": {
    "ACCOUNT": 0.0,
    "CANCEL": 0.0,
    "CONTACT": 0.0,
    "DELIVERY": 0.9466,
    "FEEDBACK": 0.0,
    "INVOICE": 0.0,
    "ORDER": 0.0533,
    "PAYMENT": 0.0,
    "REFUND": 0.0001,
    "SHIPPING": 0.0,
    "SUBSCRIPTION": 0.0
  },
  "tokens_used": 19
}
```

## Tecnologias usadas

- Python
- Flask
- Flask-CORS
- Pandas
- NLTK

## Restricciones y notas tecnicas

- El clasificador esta implementado manualmente; no usa `scikit-learn`.
- El texto del dataset esta en ingles, por lo que las mejores predicciones se obtienen con tickets redactados en ingles.
- El preprocesamiento protege negaciones utiles como `not`, `need` y `help`.
- El modelo utiliza probabilidades en logaritmos y suavizado de Laplace para evitar problemas numericos y de frecuencia cero.
