
from flask import Flask, render_template,Response, request
import numpy as np
import pickle as pk
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

app = Flask(__name__)

model = pk.load(open('model.pkl','rb'))
vectorizer = pk.load(open('vectorizer.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for letter in text:
        if letter.isalnum():
            y.append(letter)
    
    text = y[:]
    y.clear()
    for letter in text:
        if letter not in stopwords.words('english') and letter not in string.punctuation:
            y.append(letter)
    text = y[:]
    y.clear()
    for letter in text:
        y.append(ps.stem(letter))
    

    return " ".join(y)




@app.route('/')
def home():
    home = True
    return render_template('index.html', home=home)

@app.route('/results', methods=['POST','GET'])
def results():
    if request.method=='POST':
        input_text = request.form['message']
        
        transformed_sms = transform_text(input_text)

         # vectorize
        vector_input = vectorizer.transform([transformed_sms]).toarray()

        # predict
        result = model.predict(vector_input)[0]
        return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True)