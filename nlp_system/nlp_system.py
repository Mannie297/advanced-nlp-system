import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    pipeline
)
import spacy
from typing import Dict, List, Tuple, Union
import numpy as np

class AdvancedNLPSystem:
    def __init__(
        self,
        sentiment_model: str = "distilbert-base-uncased-finetuned-sst-2-english",
        ner_model: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512
    ):
        """
        Initialize the Advanced NLP System with transformer models for different tasks.
        
        Args:
            sentiment_model (str): Model name for sentiment analysis
            ner_model (str): Model name for named entity recognition
            device (str): Device to run models on ('cuda' or 'cpu')
            max_length (int): Maximum sequence length for transformer models
        """
        self.device = device
        self.max_length = max_length
        
        # Initialize tokenizers
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model)
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ner_model)
        
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=sentiment_model,
            device=0 if device == "cuda" else -1
        )
        
        # Initialize NER pipeline
        self.ner_analyzer = pipeline(
            "ner",
            model=ner_model,
            aggregation_strategy="simple",
            device=0 if device == "cuda" else -1
        )
        
        # Load spaCy for additional NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
    
    def _chunk_text(self, text: str, tokenizer) -> List[str]:
        """Split text into chunks that fit within the model's maximum sequence length."""
        tokens = tokenizer.tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for token in tokens:
            current_chunk.append(token)
            current_length += 1
            
            if current_length >= self.max_length - 2:  # -2 for [CLS] and [SEP]
                chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
                current_chunk = []
                current_length = 0
        
        if current_chunk:
            chunks.append(tokenizer.convert_tokens_to_string(current_chunk))
        
        return chunks
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze the sentiment of the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing sentiment label and confidence score
        """
        chunks = self._chunk_text(text, self.sentiment_tokenizer)
        results = []
        
        for chunk in chunks:
            result = self.sentiment_analyzer(chunk)[0]
            results.append(result)
        
        # Aggregate results
        if not results:
            return {"sentiment": "NEUTRAL", "confidence": 0.5}
        
        # Weighted average based on chunk lengths
        total_length = sum(len(chunk) for chunk in chunks)
        weighted_scores = []
        
        for chunk, result in zip(chunks, results):
            weight = len(chunk) / total_length
            score = result["score"] if result["label"] == "POSITIVE" else 1 - result["score"]
            weighted_scores.append(score * weight)
        
        final_score = sum(weighted_scores)
        return {
            "sentiment": "POSITIVE" if final_score > 0.5 else "NEGATIVE",
            "confidence": max(final_score, 1 - final_score)
        }
    
    def extract_entities(self, text: str) -> List[Dict[str, Union[str, float, List[int]]]]:
        """
        Extract named entities from the input text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            List of dictionaries containing entity information
        """
        chunks = self._chunk_text(text, self.ner_tokenizer)
        all_entities = []
        
        for chunk in chunks:
            entities = self.ner_analyzer(chunk)
            all_entities.extend(entities)
        
        return all_entities
    
    def analyze_text(self, text: str) -> Dict:
        """
        Perform comprehensive text analysis including sentiment and NER.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing all analysis results
        """
        # Get sentiment analysis
        sentiment_result = self.analyze_sentiment(text)
        
        # Get named entities
        entities = self.extract_entities(text)
        
        # Get spaCy analysis
        doc = self.nlp(text)
        
        # Extract key phrases and dependencies
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        dependencies = [(token.text, token.dep_, token.head.text) 
                       for token in doc if token.dep_ != "punct"]
        
        return {
            "sentiment": sentiment_result,
            "entities": entities,
            "key_phrases": key_phrases,
            "dependencies": dependencies
        }
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        """
        Analyze multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List of analysis results for each text
        """
        return [self.analyze_text(text) for text in texts] 