from Transformer import WordPredictor

if __name__ == "__main__":
    predictor = WordPredictor()
    
    # Try to load the existing model
    if predictor.load_saved_model():
        print("Model loaded successfully.")
        # Run predictions from the saved model
        predictor.predict_next_words()
    else:
        print("Saved model not found. Please run Transformer.py first to train the model on your data.")
