from flask import Flask, render_template, request
from scipy.special import boxcox1p
import numpy as np
import pickle
import joblib

app = Flask(__name__)

@app.route('/')
def index():
	return render_template('Details.html')

@app.route('/hello', methods=['POST'])

# def getEncoded(test_data):

#     return test_encoded_x

def hello():
	story = str(request.form['story'])
	area = request.form['area']
	street = request.form['street']
	utilities = request.form['utilities']
	neighbor = request.form['neighbor']
	bldgtype = request.form['bldgtype']
	housestyle = request.form['housestyle']
	quality = request.form['quality']
	condition = str(request.form['condition'])
	year = request.form['year']
	foundation = request.form['foundation']
	garage = request.form['garage']
	pool = request.form['pool']

	pkl_file = open('Encoder.pkl', 'rb')
	lbl = pickle.load(pkl_file)
	pkl_file.close()
	
	test = [street,condition,pool,street]
	print(test)
	x = lbl.transform(test)
	street = x[0]
	condition = x[1]
	pool = x[2]
	street = x[3]
	
	area = boxcox1p(float(area), 0.15) + 1
	story = boxcox1p(float(story), 0.15) + 1 
	condition = boxcox1p(float(condition), 0.15) + 1
	quality = boxcox1p(float(quality), 0.15) + 1
	year = boxcox1p(float(year), 0.15) + 1

	test_data = np.asarray([[area,bldgtype,foundation,garage,housestyle,neighbor,story,condition,quality,pool,street,year]])

	labelencoder_dict = joblib.load('labelencoder_dict.joblib')
	onehotencoder_dict = joblib.load('onehotencoder_dict.joblib')
	model = joblib.load('xgboost_model.joblib')
	encoded_data = None
	for i in range(0,test_data.shape[1]):
		if i in [1,2,3,4,5]:
			label_encoder =  labelencoder_dict[i]
			feature = label_encoder.transform(test_data[:,i])
			feature = feature.reshape(test_data.shape[0], 1)
			onehot_encoder = onehotencoder_dict[i]
			feature = onehot_encoder.transform(feature)
		else:
			feature = test_data[:,i].reshape(test_data.shape[0], 1)
		if encoded_data is None:
			encoded_data = feature
		else:
			encoded_data = np.concatenate((encoded_data, feature), axis=1)
	
	price = np.expm1(model.predict(encoded_data))
	print(price[0]) #This is your answer
	
	variable = price[0];
	return render_template("result.html", result = variable)

if ("__name__ == '__main__"):
	app.run(debug = True)