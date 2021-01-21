import pandas as pd
import numpy as np
import joblib

def load_model():
	model_path = 'final_model_v1.gz'
	return joblib.load(model_path)

def load_transormator():
	transformator_path = 'final_transofrmator_v1.gz'
	return joblib.load(transformator_path)

def make_test_prediction(int_features):
	model = load_model()
	transformator = load_transormator()
	
	#CREATE DATAFRAME
	cat_columns = ['fuell','transmission', 'car_model', 'car_brand', 'country_from', 'car_type']
	num_columns = ['engine_power', 'drs_ppl', 'y_m', 'year', 'milage', 'service_book', 'condition', 'airbags', 'air_condition', 'n_doors', 'n_people', 'car_category', 'extra_category']
	all_columns = ["engine_power", "drs_ppl", "y_m", "year", "milage", "service_book", "condition", "airbags", "air_condition", "n_doors", "n_people", "car_category", "extra_category", "fuell", 'transmission', 'car_model', 'car_brand', 'country_from', 'car_type']

	to_predict = pd.DataFrame(columns = all_columns)
	
	to_predict.loc[0, "car_brand"] = int_features[0]
	to_predict.loc[0, "car_model"] = int_features[1]
	to_predict.loc[0, "car_type"] = int_features[2]
	to_predict.loc[0, "engine_power"] = int_features[3]
	to_predict.loc[0, "condition"] = int_features[4]
	to_predict.loc[0, "service_book"] = int_features[5]
	to_predict.loc[0, "year"] = int_features[6]
	to_predict.loc[0, "milage"] = int_features[7]
	to_predict.loc[0, "fuell"] = int_features[8]
	to_predict.loc[0, "n_people"] = int_features[9]
	to_predict.loc[0, "airbags"] = int_features[10]
	to_predict.loc[0, "n_doors"] = int_features[11]
	to_predict.loc[0, "air_condition"] = int_features[12]
	to_predict.loc[0, "transmission"] = int_features[13]
	to_predict.loc[0, "country_from"] = int_features[14]
	to_predict.loc[0, "extra_category"] = int_features[15]
	
	
	#CAR BRAND CATEGORY
	luxury_brand = ['Jaguar','Mercedes-Benz','BMW',  'Audi', 'Volvo', 'Subaru','Porsche',
	                'Jeep','Land Rover','Cadillac','Lincoln','Infiniti',  'Ferrari','Tesla',
	                'Mini', 'Maserati', 'Aston Martin', 'Hurtan','Rover', 'Lotus','Bentley', 
	                'Lamborghini','GMC', 'Rolls-Royce', 'Alpina', 'McLaren', 'Morgan', 
	                'Maybach','Pontiac']
	
	middle_brand = ['Ford', 'Škoda','Opel', 'Toyota','Volkswagen','Peugeot', 'Suzuki',
	                'Hyundai','Dodge','Nissan','Mazda', 'DS','Honda','Chevrolet',
	               'Chrysler','Alfa Romeo', 'Saab','Lexus','Smart','Hummer','Abarth','Buick','Acura', 'Iveco',
	               'Cupra','Gonow']
	
	cheap_brand = ['Dacia','Fiat', 'Renault','Seat', 'Kia','Citroën','Mitsubishi','Microcar',
	               'Daewoo','Isuzu', 'Lancia','Ligier', 'SsangYong','Aixam','Daihatsu','Lada','Great Wall','MG',
	              'Casalini','Chatenet']
	
	#ORDINALS aircondition
	to_predict.air_condition = to_predict.air_condition.replace(['bez klimatizace'], 0)
	to_predict.air_condition = to_predict.air_condition.replace(['manuální'], 1)
	to_predict.air_condition = to_predict.air_condition.replace(['automatická'], 2)
	to_predict.air_condition = to_predict.air_condition.replace(['dvouzónová automatická'], 3)
	to_predict.air_condition = to_predict.air_condition.replace(['třízónová automatická'], 4)
	to_predict.air_condition = to_predict.air_condition.replace(['čtyřzónová automatická'], 5)
	
	#ORDINALS book
	to_predict.service_book = to_predict.service_book.replace(['ano'], 1)
	to_predict.service_book = to_predict.service_book.replace(['ne'], 0)
	to_predict.service_book = to_predict.service_book.replace([np.nan], 0)
	
	#ORDINALS condition
	to_predict.condition = to_predict.condition.replace(['nové'], 2)
	to_predict.condition = to_predict.condition.replace(['předváděcí'], 1)
	to_predict.condition = to_predict.condition.replace(['ojeté'], 0)
	
	to_predict[num_columns] = to_predict[num_columns].astype(np.float32)
	
	#adding car category depending on brand
	for index, row in to_predict.iterrows():
	    if row.car_brand in luxury_brand:
	        to_predict.loc[index,'car_category'] = 2
	    if row.car_brand in middle_brand:
	        to_predict.loc[index,'car_category'] = 1
	    if row.car_brand in cheap_brand:
	        to_predict.loc[index,'car_category'] = 0
	
	#change milage to 1, so years can be divided by it
	to_predict.loc[to_predict['milage'] == 0, 'milage'] = 1
	
	#years will only be fro 1 to 11
	to_predict.loc[to_predict['year'] != np.nan, 'year'] = to_predict['year'] - 2009
	    
	to_predict['y_m'] = to_predict.milage / to_predict.year
	to_predict['drs_ppl'] = to_predict.n_doors / to_predict.n_people

	to_predict_final = transformator.transform(to_predict[all_columns])
	prediction = model.predict(to_predict_final)

	return prediction[0]













