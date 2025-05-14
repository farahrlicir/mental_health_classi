import os
import requests
import torch
from flask import Flask, render_template, request
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

app = Flask(__name__)

# Google Drive file ID and URL
MODEL_FILE_ID = "1fuR4gL03rjiXh1zsfDy4xnO_0_XXh3HS"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}"
MODEL_PATH = "best_model_dropout02_distilbert.pth"

# Download the model if not already present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded.")

# Load tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class CustomModel(DistilBertForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(config.hidden_size, 6)
        )

model = CustomModel.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Home route
@app.route('/')
def home():
    return render_template("index.html", result="")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get("text")
    inputs = tokenizer(input_text, max_length=500, padding='max_length', truncation=True, return_tensors='pt')

    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    return render_template("index.html", result=predicted_class_id)

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
