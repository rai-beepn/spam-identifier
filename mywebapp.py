
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import os
import re
import numpy as np
from nltk.stem.porter import PorterStemmer


portstemmer = PorterStemmer()


# loading the serialized classifier, vectorizer and stopwords
classifier = pickle.load(open('bayes_classifier.pickle', 'rb'))

vectorizer = pickle.load(open('vector.pickle', 'rb'))

stop = pickle.load(open('stop.pickle', 'rb'))


def corpus_message(text):
    mess = re.sub('[^a-zA-Z]',repl = ' ',string = text)
    mess.lower()
    mess = mess.split()
    mess = [portstemmer.stem(word) for word in mess if word not in set(stop)]
    mess = ' '.join(mess)
    mess_array=  np.array([mess])
    return mess_array

def classify(message):
    label = {0: 'ham', 1: 'spam'}
    document = corpus_message(message)
    X = vectorizer.transform(document)
    y = classifier.predict(X)[0]
    proba = np.max(classifier.predict_proba(X))
    return label[y], proba



# create an instance (our app)
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    form = None
    if request.method == 'POST' and 'spam' in request.form:
        form = request.form['spam']
    return render_template('default.html', form=form)


@app.route('/results', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
        message = request.form['spam']
        y, proba = classify(message)
        return render_template('results.html',
                                content=message,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('results.html', name=name)

if __name__ == '__main__':
    app.run(port=5000,debug=True);