#!/usr/bin/env python3
"""
Topic Classification with TinyBERT Fine-tuning

This script fine-tunes TinyBERT for topic classification on 5 categories:
sport, politics, business, technology, entertainment.

MODEL DETAILS:
- Base Model: TinyBERT (Huawei Noah's Ark Lab)
- Parameters: ~14.5 million (much smaller than BART's 406M)
- Fine-tuning: Adds classification head and trains on topic data
- Advantages: Smaller size, faster inference, task-specific training
- Trade-off: Requires training data vs zero-shot capability

TRAINING APPROACH:
- Uses synthetic training data for demonstration
- Fine-tunes the classification head while optionally freezing base layers
- Implements proper train/validation split
- Saves trained model for future use

Install required dependencies: pip install transformers torch scikit-learn
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import os
import json

class TopicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TinyBERTTopicClassifier:
    def __init__(self, model_name="huawei-noah/TinyBERT_General_4L_312D"):
        """
        Initialize TinyBERT for topic classification with fine-tuning capability
        
        Args:
            model_name (str): The TinyBERT model to use as base
        """
        self.model_name = model_name
        # Handle different device types properly
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('cpu')  # Use CPU instead of MPS for better stability
        else:
            self.device = torch.device('cpu')
        
        # Define the 5 topic classes
        self.topics = ["sport", "politics", "business", "technology", "entertainment"]
        self.num_labels = len(self.topics)
        self.label_to_id = {label: idx for idx, label in enumerate(self.topics)}
        self.id_to_label = {idx: label for idx, label in enumerate(self.topics)}
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            problem_type="single_label_classification"
        )
        self.model.to(self.device)
        
        # Count model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("Model loaded successfully!")
        print(f"Model: {model_name}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Classification topics: {', '.join(self.topics)}")
        print(f"Device: {self.device}")
    
    def generate_training_data(self):
        """
        Generate expanded synthetic training data for demonstration
        In practice, you would use real labeled datasets like AG News, BBC News, etc.
        
        CURRENT LIMITATIONS:
        - Small dataset (~200 samples) causes poor generalization
        - Simple synthetic patterns don't capture real-world complexity  
        - Categories overlap in vocabulary leading to confusion
        - Need 1000+ samples per category for good performance
        
        BETTER ALTERNATIVES:
        - AG News dataset: 120k news articles in 4 categories
        - BBC News dataset: 2k articles in 5 categories  
        - Reuters dataset: 10k+ business/finance articles
        - Custom scraped data from news sites
        """
        training_data = [
            # Sport - expanded with more diverse examples
            ("Lakers defeated Warriors 120-115 in overtime thriller", "sport"),
            ("Tennis champion wins Wimbledon final in straight sets", "sport"),
            ("Olympic swimming records broken at Paris games", "sport"),
            ("Football quarterback throws winning touchdown pass", "sport"),
            ("Soccer World Cup finals draw massive television audience", "sport"),
            ("Baseball pitcher achieves perfect game milestone", "sport"),
            ("Basketball team signs star player to five year contract", "sport"),
            ("Golf tournament winner earns million dollar prize", "sport"),
            ("Marathon runner sets new world record time", "sport"),
            ("Hockey playoffs begin with exciting matchups", "sport"),
            
            # Politics - more varied political contexts
            ("President signs landmark climate change legislation", "politics"),
            ("Congressional hearing examines social media regulation", "politics"),
            ("Supreme Court ruling impacts voting rights laws", "politics"),
            ("Governor announces infrastructure spending plan", "politics"),
            ("International summit addresses global trade issues", "politics"),
            ("Senate debate focuses on healthcare reform proposals", "politics"),
            ("Mayor implements new public transportation policy", "politics"),
            ("Diplomatic negotiations seek to resolve border dispute", "politics"),
            ("Parliamentary election results show coalition government", "politics"),
            ("City council votes on affordable housing initiative", "politics"),
            
            # Business - diverse business scenarios
            ("Tech startup raises 100 million Series B funding", "business"),
            ("Manufacturing company reports quarterly profit surge", "business"),
            ("Retail chain announces store expansion across regions", "business"),
            ("Cryptocurrency exchange faces regulatory compliance issues", "business"),
            ("Pharmaceutical company receives FDA drug approval", "business"),
            ("Automotive industry shifts toward electric vehicle production", "business"),
            ("Banking sector implements new digital payment systems", "business"),
            ("Energy company invests in renewable power projects", "business"),
            ("E-commerce platform experiences record holiday sales", "business"),
            ("Real estate market shows signs of cooling trends", "business"),
            
            # Technology - varied tech topics
            ("Artificial intelligence breakthrough in medical diagnosis", "technology"),
            ("Cybersecurity firm discovers major data breach vulnerability", "technology"),
            ("Smartphone manufacturer unveils foldable display innovation", "technology"),
            ("Cloud computing service expands global server infrastructure", "technology"),
            ("Autonomous vehicle testing reaches new safety milestones", "technology"),
            ("Quantum computer achieves unprecedented processing speed", "technology"),
            ("Social network implements enhanced privacy protection features", "technology"),
            ("Robotics company develops advanced manufacturing automation", "technology"),
            ("Streaming platform launches interactive gaming content", "technology"),
            ("Blockchain technology finds applications in supply chain", "technology"),
            
            # Entertainment - broader entertainment coverage  
            ("Blockbuster superhero film dominates weekend box office", "entertainment"),
            ("Music streaming service adds exclusive artist content", "entertainment"),
            ("Television drama series wins multiple Emmy awards", "entertainment"),
            ("Video game developer announces highly anticipated sequel", "entertainment"),
            ("Film festival showcases independent documentary features", "entertainment"),
            ("Concert tour sells out arenas across multiple cities", "entertainment"),
            ("Streaming series breaks viewership records globally", "entertainment"),
            ("Celebrity chef opens new restaurant in downtown location", "entertainment"),
            ("Animation studio collaborates on international co-production", "entertainment"),
            ("Gaming tournament offers largest prize pool in history", "entertainment")
        ]
        
        # Create more sophisticated variations
        expanded_data = []
        for text, label in training_data:
            expanded_data.append((text, label))
            
            # Add paraphrased versions
            if "announces" in text:
                expanded_data.append((text.replace("announces", "reveals"), label))
            if "new" in text:
                expanded_data.append((text.replace("new", "innovative"), label))
            if "company" in text:
                expanded_data.append((text.replace("company", "corporation"), label))
            
        print(f"Generated {len(expanded_data)} training samples")
        print("WARNING: This is still a small dataset. For production use:")
        print("- Use real labeled datasets (AG News, Reuters, etc.)")
        print("- Aim for 1000+ samples per category")
        print("- Consider data augmentation techniques")
        
        texts, labels = zip(*expanded_data)
        label_ids = [self.label_to_id[label] for label in labels]
        
        return list(texts), label_ids
    
    def train_model(self, output_dir="./tinybert_topic_model", num_epochs=3):
        """
        Fine-tune TinyBERT on topic classification data
        """
        print("Generating training data...")
        texts, labels = self.generate_training_data()
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Create datasets
        train_dataset = TopicDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = TopicDataset(val_texts, val_labels, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
        )
        
        # Metrics computation
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        print("Starting training...")
        trainer.train()
        
        # Save model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", "w") as f:
            json.dump({
                "label_to_id": self.label_to_id,
                "id_to_label": self.id_to_label,
                "topics": self.topics
            }, f)
        
        print(f"Model saved to {output_dir}")
        return trainer
    
    def load_trained_model(self, model_dir="./tinybert_topic_model"):
        """
        Load a previously trained model
        """
        if not os.path.exists(model_dir):
            print(f"No trained model found at {model_dir}")
            return False
        
        print(f"Loading trained model from {model_dir}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load label mappings
        with open(f"{model_dir}/label_mappings.json", "r") as f:
            mappings = json.load(f)
            self.label_to_id = mappings["label_to_id"]
            self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}
            self.topics = mappings["topics"]
        
        self.model.to(self.device)
        print("Trained model loaded successfully!")
        return True
    
    def classify_topic(self, text):
        """
        Classify text using the trained model
        """
        self.model.eval()
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Move inputs to device
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get probabilities for each class
        probs = predictions.cpu().numpy()[0]
        predicted_class_id = np.argmax(probs)
        predicted_topic = self.id_to_label[predicted_class_id]
        confidence = probs[predicted_class_id] * 100
        
        # Create scores dictionary
        scores = {self.id_to_label[i]: prob * 100 for i, prob in enumerate(probs)}
        
        return {
            'predicted_topic': predicted_topic,
            'confidence': confidence,
            'scores': scores
        }

def main():
    """
    Main function for training and testing the topic classifier
    """
    classifier = TinyBERTTopicClassifier()
    
    # Check if trained model exists
    model_dir = "./tinybert_topic_model"
    
    if not os.path.exists(model_dir):
        print("\n" + "=" * 70)
        print("TRAINING MODE - No trained model found")
        print("=" * 70)
        
        train_choice = input("Do you want to train the model? (y/n): ").lower()
        if train_choice == 'y':
            trainer = classifier.train_model(output_dir=model_dir)
            print("\nTraining completed!")
        else:
            print("Using untrained model (results will be poor)")
    else:
        print("\n" + "=" * 70)
        print("LOADING TRAINED MODEL")
        print("=" * 70)
        classifier.load_trained_model(model_dir)
    
    # Test examples
    test_texts = [
        "The team won the match with an amazing goal in overtime",
        "The government announced new tax policies for businesses",
        "Tech company launches innovative AI-powered smartphone",
        "Box office numbers show record-breaking movie success",
        "Stock market volatility affects investor confidence"
    ]
    
    print("\n" + "=" * 70)
    print("TOPIC CLASSIFICATION - DEMO")
    print("=" * 70)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text}")
        result = classifier.classify_topic(text)
        
        print(f"Predicted Topic: {result['predicted_topic'].upper()}")
        print(f"Confidence: {result['confidence']:.1f}%")
        print("All scores:")
        
        # Sort scores for better display
        sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
        for topic, score in sorted_scores:
            emoji = "🏆" if topic == result['predicted_topic'] else "  "
            print(f"  {emoji} {topic.capitalize()}: {score:.1f}%")
    
    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Enter your own texts!")
    print("Type 'quit' to exit, 'retrain' to retrain model")
    print("=" * 70)
    
    while True:
        user_input = input("\nEnter text to classify: ").strip()
        
        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'retrain':
            print("Retraining model...")
            classifier.train_model(output_dir=model_dir)
            classifier.load_trained_model(model_dir)
            print("Model retrained and reloaded!")
            continue
        
        if user_input:
            result = classifier.classify_topic(user_input)
            
            print(f"\n📊 CLASSIFICATION RESULT:")
            print(f"Predicted Topic: {result['predicted_topic'].upper()}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print("\n📈 Detailed scores:")
            
            sorted_scores = sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)
            for topic, score in sorted_scores:
                emoji = "🏆" if topic == result['predicted_topic'] else "  "
                print(f"  {emoji} {topic.capitalize()}: {score:.1f}%")
        else:
            print("Please enter valid text.")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()