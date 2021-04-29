from flask import Flask, jsonify, render_template, request, url_for
import joblib
import os
import predict_car_price_v2 as predict_file
import json

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
	return render_template("car_price.html")

@app.route('/sitemap.xml', methods = ["GET"])
def sitemap():
	return render_template('sitemap.xml')

@app.route("/predict", methods = ["GET"])
def predict():
	json_data = request.values
	str_data = str(json_data.values)
	dict_data = "{" + str_data.split("{", 5)[1].split("}", 5)[0] + "}"
	dict_data = json.loads(dict_data)
	int_features = []
	pairs = dict_data.items()
	for key, value in pairs:
		int_features.append(value)

	prediction = predict_file.make_test_prediction(int_features)
	prediction_formated = "{:,.0f} Kč".format(prediction).replace(",", " ")
	print(prediction_formated)

	return prediction_formated

if __name__ == '__main__':
    app.run()















