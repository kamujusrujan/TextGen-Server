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

'''
MAYBE REQUIRED MODULES
gensim == 3.8.3
Keras == 2.4.3
nltk == 3.5
Keras-Preprocessing == 1.1.2
'''



'''
if not os.path.exists('tokenizer.pickle'):
	print('downloading tokenizer')
	file = requests.get('https://srv-store1.gofile.io/download/Q4DIiu/tokenizer.pickle')
	open('tokenizer.pickle','wb').write(file.content)


if not os.path.exists('model.h5'):
	print('downloading model')
	file = requests.get('https://srv-store1.gofile.io/download/qc40yD/model.h5')
	open('model.h5','wb').write(file.content)
'''



def download_url(url, save_path, chunk_size=128):
	print('downloading ... ')
	r = requests.get(url, stream=True)
	with open(save_path, 'wb') as fd:
		for chunk in r.iter_content(chunk_size=chunk_size):
			fd.write(chunk)
	


if not os.path.exists('files.zip'):
	download_url('https://sample-bucket095.s3.us-east-2.amazonaws.com/hyper.zip' , 'files.zip')
	print(os.system('file *'))
	with ZipFile('files.zip','r') as file:
		file.extractall()






app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


with open('tokenizer.pickle', 'rb') as handle:
	tokenizer = pickle.load(handle)


all_words = list(tokenizer.word_index.keys())

model = load_model('model.h5')


def get_title(text):
  test_text = text
  temp = tokenizer.texts_to_sequences([test_text])
  temp = pad_sequences(temp, maxlen = 10 , padding = 'post')
  out = model.predict(temp)
  sentence = [ np.argmax(word)  for word in out[0]]
  decoded = [ tokenizer.index_word[index]  for index in sentence if index!= 0 ]

  return decoded[1:-1]


@app.route('/random')
def random_headline():
	number_of_titles = int(request.args.get('items',10))
	response_texts = []
	for i in range(number_of_titles):
		i_response = get_title(" ".join(np.random.choice(all_words,10)))
		response_texts.append(" ".join(i_response))
	return jsonify({'articles' : response_texts})



@app.route("/generate" , methods = ['GET'])
def get_gen():
	key_words = request.args.get('terms')
	response_text = ' '.join(get_title(key_words))
	return jsonify({'article': response_text})


if __name__ == '__main__':
	app.run(port=int(os.environ.get('PORT', 8080)))

