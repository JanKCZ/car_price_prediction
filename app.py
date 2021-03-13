from flask import Flask, jsonify, render_template, request, url_for
import joblib
import os
import predict_car_price_v2 as predict_file

app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])
def home():
	return render_template("car_price.html")

@app.route("/predict", methods = ["POST"])
def predict():
	int_features = [x for x in request.form.values()]
	prediction = predict_file.make_test_prediction(int_features)
	return render_template('car_price.html',pred_text = "Odhadovaná cena:", 
											pred_price = "{:,.0f} Kč".format(prediction).replace(",", " "))

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
    app.run()















