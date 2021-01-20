
# load and use model

import pandas as pd
import numpy as np
#import copy
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import car_prices_input_list as input_list_file
import car_prices_preprocessing
from tqdm import tqdm

def load_model():
	model_path = '/Users/jankolnik/Downloads/final_model_v1.sav'
	return joblib.load(model_path)

data_frame_training_ready = pd.read_csv('/Users/jankolnik/Downloads/car_list_sauto_preprocessed_2.csv')

model = load_model()

input_data = input_list_file.Input_list()
#pipeline = car_prices_update_model.full_pipeline

def get_input(message, is_number, source_data, min_limit, max_limit):
	if source_data:
		lenght = 0
		for data in source_data: 
			print("{}. {}".format(lenght, data))
			lenght += 1

		while True:
		    try:
		       userInput = int(input(message + ": "))       
		    except ValueError:
		       print("not a number")
		       continue
		    if min_limit <= userInput <= max_limit:
		       		return source_data[userInput]
		       		break
		    else:
		       	print("Enter number between {} and {}".format(min_limit, max_limit))
		        continue
	else:
		while True:
		    try:
		       userInput = int(input(message + ": "))
		    except ValueError:
		       print("not a number")
		       continue
		    
		    if min_limit <= userInput <= max_limit:
		       		return userInput
		       		break
		    else:
		       	print("Enter number between {} and {}".format(min_limit, max_limit))
		        continue
	return userInput

"""
length = (len(input_data.car_brand_model_dict.keys()) - 1)
car_brand = get_input("Input car brand", True, list(input_data.car_brand_model_dict.keys()), 0, length)
print("output is: ", car_brand, "\n")

car_model_data = list(input_data.car_brand_model_dict.get(car_brand))
car_model = get_input("Input car model", True, car_model_data, 0, len(car_model_data) - 1)
print("output is: ", car_model, "\n")

milage = get_input("Input car milage (0 to 700000)", True, [], 0, 700_000)
year = get_input("Input car year of manufacturing (must be younger than 2010)", True, [], 2011, 2020)
engine_power = get_input("Input car power in kW", True, [], 0, 800)
fuell = get_input("Input car fuell", True, input_data.fuell_data, 0, len(input_data.fuell_data) - 1)
n_doors = get_input("Input number of doors (1 to 6)", True, [], 1, 6)
n_people = get_input("Input number of people (1 to 6)", True, [], 1, 6)
country_from = get_input("Input car country of origin", True, input_data.country_data, 0, len(input_data.country_data) - 1)
car_type = get_input("Input car type", True, input_data.car_type_data, 0, len(input_data.car_type_data) - 1)
transmission = get_input("Input car transmission", True, input_data.transmission_data, 0, len(input_data.transmission_data) - 1)
service_book = get_input("Does the car has service book?", True, input_data.service_book_data, 0, 1)
condition = get_input("What is the car condition", True, input_data.condition_data, 0, len(input_data.condition_data) - 1)
airbags = get_input("How many airbags are in the car? (1 to 14)", True, [], 0, 14)
air_condition = get_input("What is the car condition", True, input_data.air_condition_data, 0, len(input_data.air_condition_data) - 1)
extra_category = get_input("Select level of extras", True, input_data.extras_data, 0, len(input_data.extras_data) - 1)

input_data = [car_brand, car_model, milage, year, engine_power, fuell, n_doors, 
				n_people, country_from, car_type, transmission, service_book, 
				condition, airbags, air_condition]

input_data = ['Škoda', 'Octavia', 100000, 2015, 100, 'benzín', 5, 5, 'Česká republika', 'hatchback', 'manuální (6 stupňová)', 'ano', 'ojeté', 10, 'automatická', 1]
input_data_pd = pd.DataFrame(input_data, ["car_brand", "car_model", "milage", "year", "engine_power", "fuell", "n_doors", "n_people", 
	"country_from", "car_type", "transmission", "service_book", "condition", "airbags", "air_condition", "extra_category"])
"""

preprocessing_module = car_prices_preprocessing.Preprocessing()

all_columns = ["car_brand", 'fuell','transmission', 'car_model', 'country_from', 'car_type', 'engine_power', 'drs_ppl', 'y_m', 'year', 'milage', 'service_book', 'condition', 'airbags', 'air_condition', 'n_doors', 'n_people', 'car_category', 'extras_category']

all_columns_custom = ["car_brand", 'engine_power', 'drs_ppl', 'y_m', 'year', 'milage', 'service_book', 'condition', 'airbags', 'air_condition', 'n_doors', 'n_people', 'car_category', 'extras_category']
extras_columns = data_frame_training_ready.columns[-160:-1]
for column in extras_columns:
	all_columns_custom.append(column)


cat_columns = ['fuell','transmission', 'car_model', 'car_brand', 'country_from', 'car_type']
num_columns = ['engine_power', 'year', 'milage', 'airbags', 'n_doors', 'n_people', "y_m", "drs_ppl", "extras_category", "car_category"]

"""
TODO:
- udelat transformaci custom pipeline bokem ,jako prvni ,na vsechny sploupce
- tu pote pouzit na dalsi transofmraci, a to jiz pro samotne sloupce zvlast
- uniPAY: napsat Marka nebo zavoalt Katka a probrat aktualni design, a nejak si jej uz odsouhlasit
- pripravit se na zitrejsi navrh databaze :
1/ moznost propojeni promo s detailem, obe znaji IDs
2/ cerpaci stanice bude mit info o probihajicich promoakcich, jak ale zobrazit vsechny CS k dane promoakci? (zapise se k dane promoakci?)
3/ jak jinak zjistime, jaka CS ma jakou promoakci? bez stazeni vsech, anebo bez database promos / CS?

"""

y = data_frame_training_ready.pop('price').astype(np.float32)



data_frame_training_ready["y_m"] = 0
data_frame_training_ready["drs_ppl"] = 0
data_frame_training_ready["car_category"] = 0
data_frame_training_ready["extras_category"] = 0

preprocessing_pipe = make_pipeline(
		preprocessing_module
	)

full_pipeline = ColumnTransformer([
		# ("preprocess", preprocessing_pipe, all_columns_custom),
		("num", RobustScaler(), num_columns),
        ("cat_one_hot", OneHotEncoder(handle_unknown= "ignore", sparse = False), cat_columns)
        ]
        ,n_jobs = -1)

fitted = preprocessing_pipe.fit_transform(data_frame_training_ready)
# fitted[num_columns].astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(fitted, y, test_size=0.2, random_state=42)

# X_train_1 = custom_preprocessing.fit_transform(X_train.dropna())
# X_test_1 = custom_preprocessing.transform(X_train.dropna())

# X_train_final = full_pipeline.fit_transform(X_train.dropna())
# X_test_final = full_pipeline.transform(X_test.dropna())

# print("chrome NTB has 638 columns")
# print(X_train.shape)
# print(X_test_final.shape)
# print(X_test_final[1])
































