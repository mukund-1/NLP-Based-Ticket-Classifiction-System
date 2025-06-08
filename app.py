import gradio as gr
import joblib
import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re
from dateutil import parser
from textblob import TextBlob

def extract_custom_features(text_series):
    # Extracts length and sentiment
    text_length = text_series.apply(len)
    sentiment = text_series.apply(lambda x: TextBlob(x).sentiment.polarity)
    return pd.DataFrame({
        'text_length': text_length,
        'sentiment': sentiment
    })
    
 

class EntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, product_list, complaint_keywords):
        self.product_list = product_list
        self.complaint_keywords = complaint_keywords

    def extract_dates(self, text):
        dates = []
        tokens = re.findall(r'\b\w+\b', text)
        for i in range(len(tokens)):
            try:
                chunk = ' '.join(tokens[i:i+3])
                dt = parser.parse(chunk, fuzzy=False)
                dates.append(str(dt.date()))
            except:
                continue
        return list(set(dates))

    def extract_product_names(self, text):
        return [prod for prod in self.product_list if prod and prod.lower() in text.lower()]

    def extract_complaint_keywords(self, text):
        return [kw for kw in self.complaint_keywords if re.search(rf'\b{re.escape(kw)}\b', text.lower())]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # X is expected to be a pandas Series of ticket texts
        entities = X.apply(lambda text: self.extract_entities(text))
        return pd.DataFrame(entities.tolist())

    def extract_entities(self, ticket_text):
        return {
            'product_names': self.extract_product_names(ticket_text),
            'dates': self.extract_dates(ticket_text),
            'complaint_keywords': self.extract_complaint_keywords(ticket_text)
        }


# Define your custom Word2VecVectorizer class to enable pipeline deserialization
# Word2Vec vectorizer
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.vstack(X.apply(self._document_vector).values)
    
    def _document_vector(self, tokens):
        tokens = [t for t in tokens if t in self.model.wv]
        if len(tokens) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(self.model.wv[tokens], axis=0)

def preprocess_series(series):
	return series.apply(lambda x: x.lower().split())


issue_type_pipeline = joblib.load("pipeline/issue_type_pipeline.pkl")
urgency_level_pipeline = joblib.load("pipeline/urgency_level_pipeline.pkl")
entity_extraction_pipeline = joblib.load("pipeline/entity_extraction_pipeline.pkl")

# 4. Load label encoders
issue_encoder = joblib.load("models/issue_encoder.pkl")
urgency_encoder = joblib.load("models/urgency_encoder.pkl")

# 5. Prediction logic
def predict_all(text):
    df = pd.DataFrame({'ticket_text': [text]})

    # Temporary: print feature size to debug
    try:
        features = urgency_level_pipeline.named_steps['feature_union'].transform(df['ticket_text'])
        print("Feature shape at prediction time:", features.shape)
    except Exception as e:
        print("Feature extraction debug:", str(e))

    issue_pred = issue_type_pipeline.predict(df['ticket_text'])[0]
    issue = issue_encoder.inverse_transform([issue_pred])[0]

    urgency_pred = urgency_level_pipeline.predict(df['ticket_text'])[0]
    urgency = urgency_encoder.inverse_transform([urgency_pred])[0]

    entities_df = entity_extraction_pipeline.transform(df['ticket_text'])
    entities = entities_df.iloc[0].to_dict()

    result = {
        'Issue Type': '{}'.format(issue),
        'Urgency Level': '{}'.format(urgency),
        'Extracted Entities': {
            'product_names': entities['product_names'],
            'dates': entities['dates'],
            'complaint_keywords': entities['complaint_keywords']
        }
    }
    
    # Return three outputs as required
    return result['Issue Type'], result['Urgency Level'], result['Extracted Entities']


# 6. Gradio UI
demo = gr.Interface(
    fn=predict_all,
    inputs=gr.Textbox(lines=5, label="Enter Ticket Text"),
    outputs=[
        gr.Textbox(label="Predicted Issue Type"),
        gr.Textbox(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Ticket Classification and Entity Extraction",
    description="Predicts issue type and urgency level, and extracts entities like product names, dates, and complaint keywords."
)

if __name__ == "__main__":
    demo.launch()