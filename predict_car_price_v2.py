import pandas as pd
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
	num_columns = ['engine_power', 'year', 'milage', 'service_book', 'condition', 'air_condition', 'n_doors']
	all_columns = ["engine_power", "year", "milage", "service_book", "condition", "air_condition", "n_doors", "fuell", 'transmission', 'car_model', 'car_brand', 'country_from', 'car_type']

	to_predict = pd.DataFrame(columns = all_columns)
	
	to_predict.loc[0, "car_brand"] = int_features[0]
	to_predict.loc[0, "car_model"] = int_features[1]
	to_predict.loc[0, "car_type"] = int_features[2]
	to_predict.loc[0, "service_book"] = int_features[3]
	to_predict.loc[0, "condition"] = int_features[4]
	to_predict.loc[0, "year"] = int_features[5]
	to_predict.loc[0, "milage"] = int_features[6]
	to_predict.loc[0, "engine_power"] = int_features[7]
	to_predict.loc[0, "fuell"] = int_features[8]
	to_predict.loc[0, "n_doors"] = int_features[9]
	to_predict.loc[0, "air_condition"] = int_features[10]
	to_predict.loc[0, "transmission"] = int_features[11]
	to_predict.loc[0, "country_from"] = int_features[12]
	
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
	
	#ORDINALS condition
	to_predict.condition = to_predict.condition.replace(['nové'], 2)
	to_predict.condition = to_predict.condition.replace(['předváděcí'], 1)
	to_predict.condition = to_predict.condition.replace(['ojeté'], 0)
	
	to_predict[num_columns] = to_predict[num_columns].astype("float32")
	
	#years will only be fro 1 to 20
	to_predict['year'] = to_predict['year'] - 2000

	to_predict_final = transformator.transform(to_predict[all_columns])
 
	if "MLPRegressor" in str(model.__str__):
		prediction = model.predict(to_predict_final)
		return prediction[0]
	else:
		test_pred = torch.tensor(to_predict_final)
		prediction = model(test_pred.float()).item()
		return prediction
	













