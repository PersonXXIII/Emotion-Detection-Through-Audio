from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import librosa
import numpy as np
import os
import io
from sklearn.preprocessing import LabelEncoder

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_path = "model/"  # Update with your path
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Load the label encoder for emotion labels
label_encoder = LabelEncoder()
emotion_list = ['happy', 'neutral', 'sad', 'angry', 'fear', 'disgust', 'surprise']
label_encoder.fit(emotion_list)

# Function to convert audio file to features (simulated)
def extract_audio_features(audio_file):
    # Load the audio file using librosa
    y, sr = librosa.load(audio_file, sr=None)  # sr=None ensures original sampling rate is used
    # Extract features from the audio (e.g., MFCC, chroma, etc.)
    # Here, we simulate feature extraction with a placeholder
    features = f"Features of the audio file"
    return features

# Create the EmotionDataset class to process the input data
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, audio_features, tokenizer, max_length):
        self.audio_features = audio_features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_features)

    def __getitem__(self, idx):
        audio_feature = self.audio_features[idx]
        inputs = self.tokenizer(
            audio_feature,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0)
        }

# Prediction function
def predict_emotion(audio_files):
    # Convert input files to features
    audio_features = [extract_audio_features(file) for file in audio_files]
    test_dataset = EmotionDataset(audio_features, tokenizer, max_length=128)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    confidences = []
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Softmax to get the probabilities (confidence scores)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence, pred = torch.max(probabilities, dim=1)
            
            decoded_pred = label_encoder.inverse_transform([pred.item()])[0]
            predictions.append(decoded_pred)
            confidences.append(confidence.item())

    return predictions, confidences

# Define API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the request contains audio files
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        # Get the audio file(s) from the request
        audio_files = request.files.getlist('audio_file')
        
        # Save the files temporarily to process them
        file_paths = []
        for audio in audio_files:
            file_path = os.path.join('temp', audio.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            audio.save(file_path)
            file_paths.append(file_path)
        
        # Predict the emotions and their confidence
        predictions, confidences = predict_emotion(file_paths)
        
        # Clean up the temporary files
        for file_path in file_paths:
            os.remove(file_path)
        
        # Return the result as JSON, including the predictions and their confidence
        result = [{'emotion': prediction, 'confidence': confidence} for prediction, confidence in zip(predictions, confidences)]
        return jsonify({'predictions': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the API
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
