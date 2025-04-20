import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import spacy
from typing import List, Dict, Union
import numpy as np

class NLPSystem:
    def __init__(self):
        """
        Initialize the NLP system with pre-trained models for all tasks.
        """
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_sentiment_model()
        self._initialize_ner_model()
        self._initialize_classifier()

    def _initialize_sentiment_model(self):
        """Initialize the sentiment analysis model."""
        print("Initializing sentiment model...")
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model.to(self.device)
        print("Sentiment model initialized.")

    def _initialize_ner_model(self):
        """Initialize the NER model."""
        print("Initializing NER model...")
        try:
            self.ner_model = spacy.load("en_core_web_sm")
            print("NER model initialized.")
        except OSError:
            print("Downloading NER model...")
            import subprocess
            subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.ner_model = spacy.load("en_core_web_sm")
            print("NER model downloaded and initialized.")

    def _initialize_classifier(self):
        """Initialize the text classification model."""
        print("Initializing text classifier...")
        self.classifier = pipeline(
            "text-classification",
            model="bert-base-uncased",
            tokenizer="bert-base-uncased",
            device=0 if torch.cuda.is_available() else -1
        )
        print("Text classifier initialized.")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis on the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing sentiment scores
        """
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            
        return {
            "positive": scores[0][1].item(),
            "negative": scores[0][0].item()
        }

    def extract_entities(self, text: str) -> List[Dict[str, Union[str, str]]]:
        """
        Perform named entity recognition on the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List[Dict[str, Union[str, str]]]: List of dictionaries containing entity information
        """
        doc = self.ner_model(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
            
        return entities

    def classify_text(self, text: str) -> Dict[str, float]:
        """
        Perform text classification on the input text.
        
        Args:
            text (str): Input text to classify
            
        Returns:
            Dict[str, float]: Dictionary containing classification results
        """
        result = self.classifier(text)[0]
        return {
            "label": result["label"],
            "score": result["score"]
        }

    def process_text(self, text: str) -> Dict:
        """
        Process text through all three NLP tasks.
        
        Args:
            text (str): Input text to process
            
        Returns:
            Dict: Dictionary containing results from all three tasks
        """
        return {
            "sentiment": self.analyze_sentiment(text),
            "entities": self.extract_entities(text),
            "classification": self.classify_text(text)
        } 