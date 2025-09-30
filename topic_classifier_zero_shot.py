#!/usr/bin/env python3
"""
Topic Classification Script

This script classifies text into 5 categories: sport, politics, business, technology, entertainment.

MODEL DETAILS:
- Uses BART-MNLI: BART-large fine-tuned on Multi-Genre Natural Language Inference dataset
- Parameters: ~406 million (BART-large base model size)
- BART: Bidirectional and Auto-Regressive Transformers (Facebook/Meta)
- MNLI: Trained to determine text relationships (entailment, contradiction, neutral)
- Zero-shot capability: Can classify into arbitrary categories without specific training
- Much larger than DistilBERT (66M) but enables flexible topic classification

HOW IT WORKS:
- For each topic, creates hypothesis: "This text is about [topic]."
- Uses MNLI model to check if text entails the hypothesis
- Higher entailment probability = more likely the text belongs to that topic
- No training required for new topics - just define them in the topics list

Install required dependencies: pip install transformers torch
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class TopicClassifier:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initialize topic classifier using zero-shot classification
        
        Args:
            model_name (str): The model to use for classification
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Define the 5 topic classes
        self.topics = [
            "sport",
            "politics", 
            "business",
            "technology",
            "entertainment"
        ]
        
        print("Model loaded successfully!")
        print(f"Model: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Classification topics: {', '.join(self.topics)}")
    
    def classify_topic(self, text):
        """
        Classify text into one of the 5 topic categories
        
        Args:
            text (str): The text to classify
            
        Returns:
            dict: Dictionary with predicted topic and confidence scores
        """
        results = {}
        
        # For each topic, check if the text belongs to that category
        for topic in self.topics:
            # Create hypothesis for zero-shot classification
            hypothesis = f"This text is about {topic}."
            
            # Tokenize premise and hypothesis
            inputs = self.tokenizer(
                text,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                # For MNLI: 0=contradiction, 1=neutral, 2=entailment
                probs = F.softmax(outputs.logits, dim=-1)
                # We want the entailment probability (index 2)
                entailment_prob = probs[0][2].item()
                results[topic] = entailment_prob
        
        # Find the topic with highest probability
        predicted_topic = max(results, key=results.get)
        confidence = results[predicted_topic] * 100
        
        # Convert to percentages
        total_prob = sum(results.values())
        normalized_scores = {topic: (score / total_prob) * 100 for topic, score in results.items()}
        
        return {
            'predicted_topic': predicted_topic,
            'confidence': confidence,
            'scores': normalized_scores
        }

def main():
    """
    Main function to demonstrate topic classification
    """
    # Initialize the classifier
    classifier = TopicClassifier()
    
    # Example texts for each topic
    example_texts = [
        "The Lakers defeated the Warriors 120-115 in an exciting basketball game last night.",
        "The President announced new economic policies during today's press conference.",
        "Apple's stock price rose 5% after reporting better than expected quarterly earnings.",
        "Scientists developed a new AI algorithm that can process images 10 times faster.",
        "The new Marvel movie broke box office records in its opening weekend."
    ]
    
    print("=" * 70)
    print("TOPIC CLASSIFICATION - DEMO")
    print("=" * 70)
    
    # Classify example texts
    for i, text in enumerate(example_texts, 1):
        print(f"\nText {i}: {text}")
        result = classifier.classify_topic(text)
        
        print(f"Predicted Topic: {result['predicted_topic'].upper()}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print("All scores:")
        for topic, score in result['scores'].items():
            emoji = "🏆" if topic == result['predicted_topic'] else "  "
            print(f"  {emoji} {topic.capitalize()}: {score:.1f}%")
    
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Enter your own texts!")
    print("Type 'quit' to exit")
    print("=" * 70)
    
    # Interactive mode
    while True:
        user_text = input("\nEnter text to classify: ").strip()
        if user_text.lower() == 'quit':
            break
        
        if user_text:
            result = classifier.classify_topic(user_text)
            
            print(f"\n📊 CLASSIFICATION RESULT:")
            print(f"Predicted Topic: {result['predicted_topic'].upper()}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print("\n📈 Detailed scores:")
            
            # Sort scores by value for better display
            sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
            for topic, score in sorted_scores:
                emoji = "🏆" if topic == result['predicted_topic'] else "  "
                print(f"  {emoji} {topic.capitalize()}: {score:.1f}%")
        else:
            print("Please enter valid text.")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()