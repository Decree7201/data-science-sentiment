# tools.py

# First, you need to install the necessary libraries:
# pip install transformers torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def create_scoring_model():
    """
    Loads and returns a pre-trained sentiment analysis model and tokenizer.
    This model is optimized for review-style text and predicts a score from 1 to 5.
    """
    print("Loading the sentiment scoring model... (This may take a moment on first run)")
    # This model is specifically trained for sentiment analysis on reviews and outputs 1-5 stars.
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("Model loaded successfully.")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed correctly.")
        return None, None

def score_comments_tool(comments, tokenizer, model):
    """
    A "tool" function that analyzes a list of text comments and assigns a sentiment score.

    Args:
        comments (list of str): A list of user comments to analyze.
        tokenizer: The tokenizer from the pre-trained model.
        model: The pre-trained sentiment analysis model.

    Returns:
        list of dict: A list of dictionaries, each containing the original
                      comment and its assigned score. Returns None if inputs are invalid.
    """
    if not tokenizer or not model or not comments:
        print("Error: Invalid inputs provided to scoring tool.")
        return None

    results = []
    print(f"\nScoring {len(comments)} comments with the tool...")

    # Process each comment
    for i, comment in enumerate(comments):
        # Tokenize the text, ensuring it's not too long for the model
        inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True, max_length=512)

        # Get the model's prediction without calculating gradients
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # The model outputs logits for 5 classes (representing 1 to 5 stars).
        # We find the class with the highest probability (logit score).
        predicted_class_id = torch.argmax(logits, dim=1).item()

        # The model's classes are 0-indexed (0 for "1 star", 1 for "2 stars", etc.)
        # So we add 1 to convert the 0-4 index to a 1-5 score.
        score = predicted_class_id + 1

        results.append({"text": comment, "score": score})
        print(f"  - Processed comment {i+1}/{len(comments)}")

    print("Scoring complete.")
    return results

