import pickle
import pandas as pd
from preprocessor import preprocess_ticket
from naive_bayes import NaiveBayesClassifier
from evaluator import compute_metrics, print_report

# Cargar modelo
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
model = NaiveBayesClassifier()
model.from_dict(data)

# Cargar dataset
df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
df['instruction'] = df['instruction'].fillna('')

texts = df['instruction'].tolist()
labels = df['category'].tolist()

docs = [preprocess_ticket('', t) for t in texts]
preds = model.predict(docs)

classes = model.classes
result = compute_metrics(labels, preds, classes)
print_report(result, classes)