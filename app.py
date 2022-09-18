from flask import Flask, jsonify, request, render_template
import time
import pdb
from flask_cors import CORS, cross_origin
import json
# import sentenceSimilarity  
# from sentenceSimilarity import func
#from Infes
import scipy.io.wavfile as wav
from speechpy.feature import mfcc
import numpy as np
from EstimatorTest import get_feature_vector_from_mfcc
from EstimatorTest import train
from EstimatorTest import my_model
from EstimatorTest import pred_model
from werkzeug import secure_filename
import os

app = Flask(__name__)



cors = CORS(app, resources={r"./*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

app.config["DEBUG"] = True

@app.route('/', methods=['GET'])
@cross_origin()
def healthcheck():
	success = 200
	return jsonify(success)


@app.route('/EmotionDetection', methods=['POST'])
@cross_origin()
    
def predict():
    message = "hello"
    print(request.files)
    inputs1=request.files['file']
    print("**********************************")
    print(inputs1)
    message = "success"
    status = True
    output={}
    out = []
    emotion = ''
    probs = []
    try:
        emotion, probs = pred_model(inputs1)
        print("**********************************")
        print(emotion)
        print(probs)
        probs  = probs.tolist()
#         out['emotion'] = emotion
#         out['prob'] = probs
        print("*********************************")
    except Exception as e:
        err = "Not a valied file or path: "+str(e)
        message = err
        status = False
#         output['out'] = out
    output['probs'] = probs
    output['emotion'] = emotion
    output['status'] = status
    output['message'] = message

    return jsonify(output)
    
	

app.run()
