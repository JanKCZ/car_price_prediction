from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

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

all_columns = ["car_brand", "car_model", "milage", "year", "engine_power", "fuell", "n_doors", "n_people", 
	"country_from", "car_type", "transmission", "service_book", "condition", "airbags", "air_condition", "y_m", "drs_ppl", "extras_category", "car_category"]

class Preprocessing(BaseEstimator, TransformerMixin):	
	def __init__(self):
		print("")

	def fit(self, X, y=None):
		X_n = X.copy()
		print("----------------------------fit called")
		X_n = self.create_extras_category(X_n)
		X_n = self.add_car_category(X_n)
		X_n = self.create_new_features(X_n)
		X_n = self.custom_ordinal_preprocessing(X_n)
		X_n = self.drop_older_cars(X_n)
		return self

	def transform(self, X, y = None):
		X_n = X.copy()
		print("----------------------------transform called")
		X_n = self.create_extras_category(X_n)
		X_n = self.add_car_category(X_n)
		X_n = self.create_new_features(X_n)
		X_n = self.custom_ordinal_preprocessing(X_n)
		X_n = self.drop_older_cars(X_n)
		print("----------------------------finished transforming")
		return X_n


	def add_car_category(self, X):
		print("adding car category")
		for index, row in tqdm(X.iterrows()):
			if row.car_brand in luxury_brand:
			    X.loc[index,'car_category'] = 2
			if row.car_brand in middle_brand:
			    X.loc[index,'car_category'] = 1
			if row.car_brand in cheap_brand:
			    X.loc[index,'car_category'] = 0
		return X

	def create_new_features(self, X):
		X.loc[X['milage'] == 0, 'milage'] = 1
		#years will only be fro 1 to 11
		X.loc[X['year'] != np.nan, 'year'] = X['year'] - 2009
		X['y_m'] = X.milage / X.year
		X['drs_ppl'] = X.n_doors / X.n_people
		return X

	def custom_ordinal_preprocessing(self, X):
		X.air_condition = X.air_condition.replace(['bez klimatizace'], 0)
		X.air_condition = X.air_condition.replace(['manuální'], 1)
		X.air_condition = X.air_condition.replace(['automatická'], 2)
		X.air_condition = X.air_condition.replace(['dvouzónová automatická'], 3)
		X.air_condition = X.air_condition.replace(['třízónová automatická'], 4)
		X.air_condition = X.air_condition.replace(['čtyřzónová automatická'], 5)

		X.service_book = X.service_book.replace(['ano'], 1)
		X.service_book = X.service_book.replace(['ne'], 0)
		X.service_book = X.service_book.replace([np.nan], 0)

		X.condition = X.condition.replace(['nové'], 2)
		X.condition = X.condition.replace(['předváděcí'], 1)
		X.condition = X.condition.replace(['ojeté'], 0)
		return X

	def create_extras_category(self, X):
		extras = X.columns[-164:-4]
		print("reducing extras")
		for index, row in tqdm(X.iterrows()):
			total_extras = row[extras].sum()
			if 0 <= total_extras <= 14:
			    X.loc[index,'extras_category'] = 1
			if 15 <= total_extras <= 29:
			    X.loc[index,'extras_category'] = 2
			if 30 <= total_extras <= 44:
			    X.loc[index,'extras_category'] = 3
			if 45 <= total_extras <= 59:
			    X.loc[index,'extras_category'] = 4
			if 60 <= total_extras <= 74:
			    X.loc[index,'extras_category'] = 5
			if 75 <= total_extras <= 89:
			    X.loc[index,'extras_category'] = 6
			if 90 <= total_extras <= 104:
			    X.loc[index,'extras_category'] = 7
			if 105 <= total_extras >= 119:
			    X.loc[index,'extras_category'] = 8
			if 120 <= total_extras >= 134:
			    X.loc[index,'extras_category'] = 9
			if total_extras >= 135:
			    X.loc[index,'extras_category'] = 10 
		return X

	def drop_older_cars(self, X):
		data_older_cars = X[X.year < 2011].index
		X.drop(data_older_cars)
		return X





























        

