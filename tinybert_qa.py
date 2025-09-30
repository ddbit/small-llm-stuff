#!/usr/bin/env python3
"""
TinyBERT Question Answering Script

This script uses TinyBERT model for question answering tasks.
Install required dependencies: pip install transformers torch
"""

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class TinyBERTQA:
    def __init__(self, model_name="distilbert-base-cased-distilled-squad"):
        """
        Initialize a BERT-based model for question answering
        Using DistilBERT fine-tuned on SQuAD for better QA performance
        
        Args:
            model_name (str): The model to use
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        print("Model loaded successfully!")
    
    def answer_question(self, question, context):
        """
        Answer a question based on the given context
        
        Args:
            question (str): The question to answer
            context (str): The context containing the answer
            
        Returns:
            str: The extracted answer
        """
        # Tokenize the question and context
        inputs = self.tokenizer(
            question,
            context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            start_scores = outputs.start_logits
            end_scores = outputs.end_logits
        
        # Get the most likely start and end positions
        start_idx = torch.argmax(start_scores).item()
        end_idx = torch.argmax(end_scores).item()
        
        # Ensure end_idx is after start_idx
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract the answer tokens and convert to string
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # If answer is empty or just whitespace, return "No answer found"
        if not answer.strip():
            return "No answer found"
            
        return answer.strip()

def main():
    """
    Main function to demonstrate TinyBERT question answering
    """
    # Initialize the model
    qa_model = TinyBERTQA()
    
    # Example context and questions
    context_ = """
    TinyBERT is a smaller and faster version of BERT that maintains competitive performance 
    while being much more efficient. It was developed by Huawei Noah's Ark Lab. 
    TinyBERT uses knowledge distillation to compress the original BERT model, 
    reducing its size by 7.5x and increasing inference speed by 9.4x while retaining 
    96.8% of BERT's performance on GLUE tasks.
    """
    
    context = """
On Thursday, Trump signed an order on "domestic terrorism and political violence", saying it would be used to investigate "wealthy people" who fund "professional anarchists and agitators". He suggested liberal billionaires George Soros and LinkedIn founder Reid Hoffman could be among them.

Then hours later, Trump's Justice Department announced it had indicted James Comey, the former FBI director and Trump critic whom the president had said was "guilty as hell" days earlier.

Trump has justified a looming crackdown on left-wing groups by pointing to two recent, and shocking, acts of violence. First, the killing of conservative activist Charlie Kirk on a college campus, and then this week's gun attack targeting immigration agents in Dallas, in which two migrant detainees were wounded and one killed.
"""

    questions = []
    
    q_ = [
        "Who developed TinyBERT?",
        "What technique does TinyBERT use for compression?",
        "How much faster is TinyBERT compared to BERT?"
    ]
    
    print("=" * 60)
    print("TinyBERT Question Answering Demo")
    print("=" * 60)
    print(f"Context: {context}")
    print("=" * 60)
    
    # Answer each question
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        answer = qa_model.answer_question(question, context)
        print(f"Answer: {answer}")
    
    print("\n" + "=" * 60)
    print("Interactive Mode - Enter your own questions!")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    # Interactive mode
    while True:
        user_question = input("\nEnter your question: ").strip()
        if user_question.lower() == 'quit':
            break
        
        if user_question:
            answer = qa_model.answer_question(user_question, context)
            print(f"Answer: {answer}")
        else:
            print("Please enter a valid question.")
    
    print("Goodbye!")

if __name__ == "__main__":
    main()