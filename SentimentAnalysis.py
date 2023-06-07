from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from flask import Flask, render_template, request

# Load the sentiment analysis model
Model = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(Model)
model = AutoModelForSequenceClassification.from_pretrained(Model)

# Create a Flask application
app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    # Get the input text from the form
    text = request.form['text']

    # Perform sentiment analysis on the input text
    encoded_text = tokenizer(text, return_tensors = "pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
            "Negative" : scores[0],
            "Nuetral" : scores[1],
            "Positive" : scores[2]
    }

    Keymax = max(zip(scores_dict.values(), scores_dict.keys()))[1]
    valuemax = max(zip(scores_dict.values(), scores_dict.keys()))[0]

    # Get the sentiment label and score
    label = f"The entered text is {Keymax}"
    score = valuemax

    # Render the result template with the sentiment label and score
    return render_template('result.html', label=label, score=score)

if __name__ == '__main__':
    app.run(debug=True)