from transformers import pipeline
import gradio as gr

# Load pre-trained sentiment analysis model from Hugging Face
sentiment_model = pipeline("sentiment-analysis")

# Function to predict sentiment
def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    sentiment = result['label']
    confidence = round(result['score'] * 100, 2)
    return f"Sentiment: {sentiment}, Confidence: {confidence}%"

# Gradio Interface
iface = gr.Interface(fn=analyze_sentiment, 
                     inputs="text", 
                     outputs="text", 
                     title="Sentiment Analysis",
                     description="Enter text to predict its sentiment.")

# Launch the app
iface.launch()
