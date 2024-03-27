from flask import Flask, render_template, request
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
model_name_or_path = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model.load_state_dict(torch.load('./app/model/sft_alpaca.pt'))

# Move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if request.method == 'POST':
        # Get user input from the form
        prompt = request.form['prompt']

        # Encode prompt and generate continuation
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        output = model.generate(input_ids, max_length=256, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)

        # Decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Pass the generated text back to the frontend
        return render_template('index.html', prompt=prompt, generated_text=generated_text)

if __name__ == '__main__':
    app.run(debug=True)
