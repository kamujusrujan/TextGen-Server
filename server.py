import tensorflow as tf 
import numpy as np
# import nltk 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tensorflow.keras.models import load_model
import pickle 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from warnings import filterwarnings
filterwarnings("ignore")
from flask import Flask, abort, jsonify, request
from flask_cors import CORS, cross_origin
import requests
from zipfile import ZipFile
from utils import *


'''
MAYBE REQUIRED MODULES
gensim == 3.8.3
Keras == 2.4.3
nltk == 3.5
Keras-Preprocessing == 1.1.2
'''



def download_url(url, save_path, chunk_size=128):
    print('downloading ... ')
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    print('Completed downloading . . . ')
    


if not os.path.exists('files.zip'):
    download_url('https://publicfilesyans.s3.ap-south-1.amazonaws.com/files.zip' , 'files.zip')
    with ZipFile('files.zip','r') as file:
        file.extractall()




app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

all_words = list(tokenizer.word_index.keys())
model = load_model('hyper.h5')


def get_title(text):
  test_text = text
  temp = tokenizer.texts_to_sequences([test_text])
  temp = pad_sequences(temp, maxlen = 10 , padding = 'post')
  out = model.predict(temp)
  sentence = ['sos']
  for word in out[0]:
    w = tokenizer.index_word[np.argmax(word)]
    if w == 'eos' : break
    if w != sentence[-1] : sentence.append(w)
  return sentence[1:]

''' 
  test_text = text
  temp = tokenizer.texts_to_sequences([test_text])
  temp = pad_sequences(temp, maxlen = 10 , padding = 'post')
  out = model.predict(temp)
  sentence = [ np.argmax(word)  for word in out[0]]
  decoded = [ tokenizer.index_word[index]  for index in sentence if index!= 0 ]
  return decoded[1:-1] 
'''

@app.route('/random')
def random_headline():
    number_of_titles = int(request.args.get('items',10))
    response_texts = []
    for i in range(number_of_titles):
        key_words = request.args.get('terms' ," ".join(np.random.choice(all_words,10)))
        generated_news = get_title(key_words)
        response_data = Article(title = ' '.join(generated_news),key_words =  generated_news) 
        response_texts.append(response_data.get_response())
    return jsonify({'articles' : response_texts})



@app.route("/generate" , methods = ['GET'])
def get_gen():
    key_words = request.args.get('terms' ," ".join(np.random.choice(all_words,10)))
    generated_news = get_title(key_words)
    response_data = Article(title = ' '.join(generated_news), key_words =  generated_news) 
    return jsonify({'article': response_data.get_response()})


if __name__ == '__main__':
    port=int((os.environ.get('PORT', 5000)))
    print(port) 
    app.run( host='0.0.0.0' , port=port ,debug= True)

