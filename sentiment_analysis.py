#!/usr/bin/env python3
"""
Sentiment Analysis Script

Questo script analizza il sentiment di un testo e restituisce se è positivo o negativo con percentuali.
Installa le dipendenze: pip install transformers torch
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Inizializza il modello per l'analisi del sentiment
        
        Args:
            model_name (str): Il modello da utilizzare per l'analisi del sentiment
        """
        print(f"Caricamento {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Labels per il modello twitter-roberta
        self.labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
        print("Modello caricato con successo!")
    
    def analyze_sentiment(self, text):
        """
        Analizza il sentiment del testo
        
        Args:
            text (str): Il testo da analizzare
            
        Returns:
            dict: Dizionario con sentiment e percentuali
        """
        # Tokenizza il testo
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Predizione
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
        
        # Estrai le percentuali
        scores = predictions[0].tolist()
        
        # Trova il sentiment principale
        max_score_idx = torch.argmax(predictions[0]).item()
        sentiment = self.labels[max_score_idx]
        
        # Crea il risultato
        result = {
            'sentiment': sentiment,
            'confidence': scores[max_score_idx] * 100,
            'scores': {
                'negative': scores[0] * 100,
                'neutral': scores[1] * 100,
                'positive': scores[2] * 100
            }
        }
        
        return result

def main():
    """
    Funzione principale per dimostrare l'analisi del sentiment
    """
    # Inizializza l'analizzatore
    analyzer = SentimentAnalyzer()
    
    # Example texts
    example_texts = [
        "I love this product, it's amazing!",
        "This movie is terrible, I don't recommend it to anyone.",
        "The weather today is cloudy.",
        "I'm very happy with this result!",
        "What a horrible day, everything is going wrong."
    ]
    
    print("=" * 70)
    print("ANALISI DEL SENTIMENT - DEMO")
    print("=" * 70)
    
    # Analyze example texts
    for i, text in enumerate(example_texts, 1):
        print(f"\nText {i}: {text}")
        result = analyzer.analyze_sentiment(text)
        
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print("Details:")
        print(f"  • Negative: {result['scores']['negative']:.1f}%")
        print(f"  • Neutral: {result['scores']['neutral']:.1f}%")
        print(f"  • Positive: {result['scores']['positive']:.1f}%")
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Enter your own texts!")
    print("Type 'quit' to exit")
    print("=" * 70)
    
    # Interactive mode
    while True:
        user_text = input("\nEnter text to analyze: ").strip()
        if user_text.lower() == 'quit':
            break
        
        if user_text:
            result = analyzer.analyze_sentiment(user_text)
            
            print(f"\n📊 ANALYSIS RESULT:")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print("\n📈 Detailed scores:")
            print(f"  🔴 Negative: {result['scores']['negative']:.1f}%")
            print(f"  ⚪ Neutral: {result['scores']['neutral']:.1f}%")
            print(f"  🟢 Positive: {result['scores']['positive']:.1f}%")
        else:
            print("Please enter valid text.")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()