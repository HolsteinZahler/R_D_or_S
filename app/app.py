from flask import Flask, jsonify, render_template, redirect, make_response, json, request
import os

from joblib import load

# Tools to remove stopwords from tweets
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



app = Flask(__name__)
app.config['DEBUG']= True

stop_words = set(stopwords.words('english'))

bigram_vectorizer = load('data_preprocessors/bigram_vectorizer.joblib')
bigram_tf_idf_transformer = load('data_preprocessors/bigram_tf_idf_transformer.joblib')
sgd_classifier = load('classifiers/sgd_classifier.joblib')

print('loaded everything')

@app.route('/')
def home_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def home():
    print('attempting to start template.')
    in_text = request.form['text']
    # if loading_error:
    #     def list_tostring(input_list):
    #         return '   '.join(input_list)
    #     party_result = os.getcwd()+' loaded '+str(num_loaded)+'  '+list_tostring(os.listdir())
    # else: 
    def list_tostring(input_list):
        return ' '.join(input_list)
    def remove_stopwords(input_list):
        return [w for w in input_list if not w in stop_words]
    fun_input = list_tostring(remove_stopwords(word_tokenize(in_text)))

    X_pred = bigram_vectorizer.transform([fun_input])
    X_pred = bigram_tf_idf_transformer.transform(X_pred)
    result = sgd_classifier.predict(X_pred)
    dec = sgd_classifier.decision_function(X_pred)
    if result[0] == 'R':
        party_result = 'Predicted Republican Tweet'
    elif result[0] == 'D':
        party_result = 'Predicted Democrat Tweet'
    else:
        party_result = 'Predicted Sesame Street Tweet'
    in_text = 'Inputted tweet:  '+in_text
    print('attempting to load template.')
    return render_template('index.html', tweet_string_out = in_text, result_string_out = party_result)


if __name__ == '__main__':

    app.run(debug=True)