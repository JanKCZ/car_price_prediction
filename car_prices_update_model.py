#!/usr/bin/env python
# coding: utf-8

# # Purpose of this file
# ### process new data from .csv file and output updated model, with predict and score methods

import pandas as pd
import numpy as np
import copy
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder, Normalizer, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.compose import ColumnTransformer
import sklearn
import math
from sklearn.neural_network import MLPRegressor
# from tensorflow.keras import layers
# import tensorflow as tf
from tqdm import tqdm
import joblib
from datetime import datetime


raw_data = pd.read_csv('/Users/jankolnik/Downloads/car_list_all_v1_updated_sauto.csv')

raw_data_update = pd.read_csv("/Users/jankolnik/Downloads/car_list_all_v1_sauto_update.csv")

raw_data_updated = pd.concat([raw_data, raw_data_update])

#drop duplicated adds
raw_data_updated = raw_data_updated.drop_duplicates(subset=['add_id-href'])

#save to CSV
raw_data_updated.to_csv ('/Users/jankolnik/Downloads/car_list_all_v1_updated_sauto.csv', index = False, header=True)

print("{} shape before update".format(raw_data.shape[0]))
print("added {} rows of raw data".format(raw_data_update.shape[0]))
print("final raw data shape is {}".format(raw_data_updated.shape))

#remove adds, which include words about damaged or non-functional cars
bad_words = [" vadn", " rozbit", " havarovan", " poškozen", " špatn"]
["bez poškození",  ]
bad_index = []

for word in bad_words:
    bad_index_1 = raw_data_updated[raw_data_updated.detail.str.contains(word, case = False)].index
    for index in bad_index_1:
        bad_index.append(index)

bad_index = list(set(bad_index))

pd.set_option('display.max_colwidth', None)
print("dropping wrong adds with words: ", bad_words)
print(raw_data.loc[bad_index]["detail-href"])

#dropping useless columns:
data_no_trash_columns = raw_data_updated.drop(['web-scraper-order', 'web-scraper-start-url', 
                                              'detail-href', 'detail', 
                                              'in_usage_from_year', 'additional_info', 
                                              'price_info', 'gps', 'gps-href', 'warranty_until', 'euro_certification',
                                              'first_owner', 'vin', 'add_id', 'add_id-href', 'seller_name'], axis = 1)

# manually entered extras from web filter
extras_ids_from_filters = ["ABS", "adaptivní tempomat", "asistent rozjezdu do kopce", "bezklíčkové ovládání", "bi-xenony", "bluetooth", 
"centrální zamykání", "dálkové centrální zamykání", "el. ovládání oken", "el. ovládaný kufr", "ESP", 
"hlídání mrtvého úhlu", "isofix", "kožené čalounění", "LED světlomety plnohodnotné", "litá kola", 
"nezávislé topení", "odvětrávání sedadel", "palubní počítač", "panoramatická střecha", "parkovací asistent",
"parkovací kamera", "parkovací senzory", "posilovač řízení", "satelitní navigace", "senzor stěračů",
"sledování únavy řidiče", "Start/Stop systém", "střešní okno", "tažné zařízení", "tempomat", "USB", "vyhřívaná sedadla",
"vyhřívané čelní sklo", "vyhřívaný volant", "xenony", "záruka"]

def get_extras_from(cell):
    # parse extras from cells from all car adverts.
    parse_1 = cell.replace("[", "").replace("]", "").replace("{\"extras_list\":\"", "").replace("\"}", "")
    extras_new = parse_1.split(",")

    for extra in extras_new:
        if extra not in extras_ids_from_filters:
            extras_ids_from_filters.append(extra)

#add extras from car adverts to manually entered extras
def extract_all_cells_with_extra(data):
    for _, row in data.iterrows():
        get_extras_from(row["extras_list"])

extract_all_cells_with_extra(data_no_trash_columns)

extras_ids_from_filters.pop(0)
extras_ids_from_filters.sort()
extras_ids = extras_ids_from_filters.copy()

#for each extra item create empty column with zeros
zero_matrix = np.zeros((len(data_no_trash_columns), len(extras_ids)))
extra_features_frame = pd.DataFrame(zero_matrix, index = None, columns = extras_ids)

data_no_trash_columns = pd.concat([data_no_trash_columns, extra_features_frame.reindex(data_no_trash_columns.index)], axis = 1)
data_frame_extras = data_no_trash_columns.copy()
print("create column for each extra")
for index, row in tqdm(data_frame_extras.iterrows()):
    row_content = np.str(row["extras_list"])
    for extra_id_column in extras_ids:
        if extra_id_column in row_content:
            data_frame_extras.at[index, extra_id_column] = 1

# setting pandas to use inf values as NA
pd.set_option('mode.use_inf_as_na', True) 

data_frame_number_clean = data_frame_extras.copy()

#milage
data_frame_number_clean.milage = data_frame_number_clean.milage.map(lambda x: np.str(x).replace(" ", '').replace("\xa0", "").replace("mil", "").replace("km", "").strip())

#price
data_frame_number_clean.price = data_frame_number_clean.price.map(lambda x: np.str(x).replace(" ", '').replace("\xa0", "").strip())

#ccm
data_frame_number_clean.ccm = data_frame_number_clean.ccm.map(lambda x: np.str(x).replace(" ", '').replace("\xa0", "").replace('ccm', "").strip())

#engine_power
data_frame_number_clean.engine_power = data_frame_number_clean.engine_power.map(lambda x: np.str(x).replace(" ", '').replace("\xa0", "").split("kW")[0].strip())

#year
data_frame_number_clean.year = data_frame_number_clean.year.map(lambda x: np.str(x).split("/")[-1].replace(" ", '').replace("\xa0", "").strip())

#stk
data_frame_number_clean.stk = data_frame_number_clean.stk.map(lambda x: np.str(x).replace(" ", '').replace("\xa0", "").split("/")[-1].strip())
data_frame_number_clean.milage = pd.to_numeric(data_frame_number_clean.milage, errors='coerce')
data_frame_number_clean.price = pd.to_numeric(data_frame_number_clean.price, errors='coerce')
data_frame_number_clean.engine_power = pd.to_numeric(data_frame_number_clean.engine_power, errors='coerce')
data_frame_number_clean.year = pd.to_numeric(data_frame_number_clean.year, errors='coerce')
data_frame_number_clean.stk = pd.to_numeric(data_frame_number_clean.stk, errors='coerce')
data_frame_number_clean.ccm = pd.to_numeric(data_frame_number_clean.ccm, errors='coerce')

data_frame_no_dupl = data_frame_number_clean.copy()

data_frame_no_dupl = data_frame_no_dupl.drop_duplicates(subset=['milage', 'year', 'price', 'engine_power'])
data_frame_no_dupl = data_frame_no_dupl.replace([np.inf, -np.inf], np.nan)

print("dropping cars for less than 1000 Kc and oveer 5 mil Kc")
data_price_more_5m = data_frame_no_dupl[lambda data: data.price > 5000000].index
data_price_less_1k = data_frame_no_dupl[lambda data: data.price <= 1000].index
data_frame_no_dupl = data_frame_no_dupl.drop(data_price_more_5m)
data_frame_no_dupl = data_frame_no_dupl.drop(data_price_less_1k)

print("dropping cars with power less than 5kW and more than 700kW")
data_wrong_power = data_frame_no_dupl[lambda data: data.engine_power > 700].index
data_frame_no_dupl = data_frame_no_dupl.drop(data_wrong_power)
data_wrong_power = data_frame_no_dupl[lambda data: data.engine_power < 5].index
data_frame_no_dupl = data_frame_no_dupl.drop(data_wrong_power)

print("droping cars older than 2020 with 0 milage")
data_2020 = data_frame_no_dupl[lambda data: data.year < 2020]
data_wrong_milage_indexes = data_2020[lambda data: data.milage <= 1].index
data_frame_no_dupl = data_frame_no_dupl.drop(data_wrong_milage_indexes)

print("dropping cars with milage over 700k")
data_milage_over_700k = data_frame_no_dupl[lambda data: data.milage > 700000].index
data_frame_no_dupl = data_frame_no_dupl.drop(data_milage_over_700k)

print("dropping cars with ccm over 8k")
data_ccm_more_8k = data_frame_no_dupl[lambda data: data.ccm > 8000].index
data_frame_no_dupl = data_frame_no_dupl.drop(data_ccm_more_8k)
data_frame_no_neuvedeno = data_frame_no_dupl.copy()
for column in data_frame_no_neuvedeno.columns:
    data_frame_no_neuvedeno.loc[(data_frame_no_neuvedeno[column] == 'neuvedeno'), column] = np.nan

#dropping rows without any extras, which is impossible
print("dropping rows with zero extras")
no_extras = 0
for index, row in tqdm(data_frame_no_neuvedeno.iterrows()):
        if row[extras_ids].sum() == 0:
            data_frame_no_neuvedeno = data_frame_no_neuvedeno.drop(index)

data_frame_training_ready = data_frame_no_neuvedeno.copy()

print("saving to CSV, preprocessed, shape: ", data_frame_training_ready.shape)
data_frame_training_ready.to_csv ('/Users/jankolnik/Downloads/car_list_sauto_preprocessed_2.csv', index = False, header=True)
drop_nan = True               # use KNN imputer to impute num and cat values
ccm_power = True              # new feature, ccm / power
use_ordinal = True            # use ordinal values for  air_condition, service_book, car condition
drop_year = 2011              # drop old cars with unreliable data
reduce_extras = True          # group extras to 10 options
create_features = True        # creates some features combinating older ones and drops the old ones
add_category = True           # add car category
setup_for_NN = True
drop_wrong_power_ccm_combination = True

if setup_for_NN:
  is_sparse = False              # use sparse False for NN
  handle_skewed = False         # use np.log for better distribution
else:
  is_sparse = True
  handle_skewed = True

extra_columns_random = data_frame_training_ready.columns[-162:-2]

# drop old cars with unreliable data
print("dropping older cars than: ", drop_year)
data_older_cars = data_frame_training_ready[lambda data: data.year < drop_year].index
data_frame_training_ready = data_frame_training_ready.drop(data_older_cars)

print("reducing extras to 10 categories")
if reduce_extras:
        for index, row in tqdm(data_frame_training_ready.iterrows()):
            total_extras = row[extra_columns_random].sum()
            
            if 0 <= total_extras <= 14:
                data_frame_training_ready.loc[index,'extra_category'] = 10
            if 15 <= total_extras <= 29:
                data_frame_training_ready.loc[index,'extra_category'] = 9
            if 30 <= total_extras <= 44:
                data_frame_training_ready.loc[index,'extra_category'] = 8
            if 45 <= total_extras <= 59:
                data_frame_training_ready.loc[index,'extra_category'] = 7
            if 60 <= total_extras <= 74:
                data_frame_training_ready.loc[index,'extra_category'] = 6
            if 75 <= total_extras <= 89:
                data_frame_training_ready.loc[index,'extra_category'] = 5
            if 90 <= total_extras <= 104:
                data_frame_training_ready.loc[index,'extra_category'] = 4
            if 105 <= total_extras >= 119:
                data_frame_training_ready.loc[index,'extra_category'] = 3
            if 120 <= total_extras >= 134:
                data_frame_training_ready.loc[index,'extra_category'] = 2
            if total_extras >= 135:
                data_frame_training_ready.loc[index,'extra_category'] = 1

print("adding car_category to each row")
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

if add_category:
  print("addding car category: ")
  for index, row in tqdm(data_frame_training_ready.iterrows()):
      if row.car_brand in luxury_brand:
          data_frame_training_ready.loc[index,'car_category'] = 2
      if row.car_brand in middle_brand:
          data_frame_training_ready.loc[index,'car_category'] = 1
      if row.car_brand in cheap_brand:
          data_frame_training_ready.loc[index,'car_category'] = 0

data_frame_training_ready = data_frame_training_ready.drop(data_frame_training_ready[lambda data: data.car_brand == 'Ostatní'].index)

if create_features:
    #change milage to 1, so years can be divided by it
    data_frame_training_ready.loc[data_frame_training_ready['milage'] == 0, 'milage'] = 1

    #years will only be fro 1 to 11
    data_frame_training_ready.loc[data_frame_training_ready['year'] != np.nan, 'year'] = data_frame_training_ready['year'] - 2009
        
    # num_columns = num_columns + ['pcm', 'drs_ppl', 'air_ppl', 'y_m']
    data_frame_training_ready['pcm'] = data_frame_training_ready.engine_power / data_frame_training_ready.ccm
    data_frame_training_ready['y_m'] = data_frame_training_ready.milage / data_frame_training_ready.year
    data_frame_training_ready['drs_ppl'] = data_frame_training_ready.n_doors / data_frame_training_ready.n_people
    data_frame_training_ready['air_ppl'] = data_frame_training_ready.n_people / data_frame_training_ready.airbags 



if use_ordinal:
        data_frame_training_ready.air_condition = data_frame_training_ready.air_condition.replace(['bez klimatizace'], 0)
        data_frame_training_ready.air_condition = data_frame_training_ready.air_condition.replace(['manuální'], 1)
        data_frame_training_ready.air_condition = data_frame_training_ready.air_condition.replace(['automatická'], 2)
        data_frame_training_ready.air_condition = data_frame_training_ready.air_condition.replace(['dvouzónová automatická'], 3)
        data_frame_training_ready.air_condition = data_frame_training_ready.air_condition.replace(['třízónová automatická'], 4)
        data_frame_training_ready.air_condition = data_frame_training_ready.air_condition.replace(['čtyřzónová automatická'], 5)

        data_frame_training_ready.service_book = data_frame_training_ready.service_book.replace(['ano'], 1)
        data_frame_training_ready.service_book = data_frame_training_ready.service_book.replace(['ne'], 0)
        data_frame_training_ready.service_book = data_frame_training_ready.service_book.replace([np.nan], 0)

        data_frame_training_ready.condition = data_frame_training_ready.condition.replace(['nové'], 2)
        data_frame_training_ready.condition = data_frame_training_ready.condition.replace(['předváděcí'], 1)
        data_frame_training_ready.condition = data_frame_training_ready.condition.replace(['ojeté'], 0)


cat_columns = ['fuell','transmission','car_type', 'car_model', 'condition', 'country_from', 'air_condition', 'service_book', 'car_brand']
num_columns = ['year', 'milage', 'engine_power', 'ccm', 'airbags', 'n_doors', 'n_people'] 
cat_columns_ordinal = []

if use_ordinal:
  cat_columns = ['fuell','transmission', 'car_model', 'car_brand', 'country_from', 'car_type']
  num_columns = ['year', 'milage', 'engine_power', 'ccm', 'airbags', 'n_doors', 'n_people', 'air_condition', 
                       'service_book', 'condition']
  if create_features:
    num_columns = ['engine_power', 'drs_ppl', 'y_m', 'year', 'milage', 'service_book', 'condition', 'airbags', 'air_condition', 'n_doors', 'n_people']

if add_category:
  num_columns.append('car_category')

if reduce_extras:
  num_columns.append('extra_category')
else:
  for column in ordered_extra_columns:
    num_columns.append(column)


if drop_wrong_power_ccm_combination:
  small_ccm = data_frame_training_ready[lambda data: data.ccm < 580]
  small_power_to_ccm = small_ccm[lambda data: data.engine_power > 10].index

  big_ccm = data_frame_training_ready[lambda data: data.ccm < 3500]
  big_power_to_ccm = big_ccm[lambda data: data.engine_power > 450].index
  data_frame_training_ready = data_frame_training_ready.drop(big_power_to_ccm)

  cmm_1000 = data_frame_training_ready[lambda data: data.ccm > 1000]
  ccm_1000_power_20 = cmm_1000[lambda data: data.engine_power < 20].index
  data_frame_training_ready = data_frame_training_ready.drop(ccm_1000_power_20)

important_columns = num_columns + cat_columns

if drop_nan:
  data_frame_training_ready_no_nan = data_frame_training_ready[important_columns + ['price']].dropna()
else:
  data_frame_training_ready_no_nan = data_frame_training_ready[important_columns + ['price']]

print("")
print("final number of rows before transforming: ", data_frame_training_ready_no_nan.shape[0])
print("")

data_frame_training_ready_no_nan[cat_columns] = data_frame_training_ready_no_nan[cat_columns].astype(np.str)
if handle_skewed:
  data_frame_training_ready_no_nan[num_columns] = data_frame_training_ready_no_nan[num_columns].astype(np.float32)
  data_frame_training_ready_no_nan['price'] = data_frame_training_ready_no_nan['price'].astype(np.float32)

y = data_frame_training_ready_no_nan.pop('price').astype(np.float32)
X = data_frame_training_ready_no_nan

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_cat = X_train[cat_columns]
X_train_num = X_train[num_columns]

X_test_cat = X_test[cat_columns]
X_test_num = X_test[num_columns]

num_pipeline = make_pipeline(
            RobustScaler()
        )

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_columns),
        ("cat_one_hot", OneHotEncoder(handle_unknown= "ignore", sparse = is_sparse), cat_columns)
        ]
        ,n_jobs = -1)

#-------------
#-------------START
#-------------
class InputData:
  def __init__(self, car_brand, car_model, car_type, engine_power, ccm, condition, service_book, year, milage, fuell, n_ppl, n_airbags, n_doors, 
    aircondition, transmission, country, extras):

    self.car_brand = car_brand
    self.car_model = car_model
    self.car_type = car_type
    self.engine_power = engine_power
    self.ccm = ccm
    self.condition = condition
    self.service_book = service_book
    self.year = year
    self.milage = milage
    self.fuell = fuell
    self.n_ppl = n_ppl
    self.n_airbags = n_airbags
    self.n_doors = n_doors
    self.aircondition = aircondition
    self.transmission = transmission
    self.country = country
    self.extras = extras

#TEST SAMPLE:
test_car = InputData("Škoda", "Octavia", "hatchback", "110", 1968, "ojeté", "ano", "2017", "90000", "nafta", "5", "5", "8",
  "manuální", "automatická (7 stupňová)", "Česká republika", "6")

#CREATE DATAFRAME
cat_columns_2 = ['fuell','transmission', 'car_model', 'car_brand', 'country_from', 'car_type']
num_columns_2 = ['engine_power', 'drs_ppl', 'y_m', 'year', 'milage', 'service_book', 'condition', 'airbags', 'air_condition', 'n_doors', 'n_people', 'extra_category', 'car_category']
all_columns = cat_columns_2 + num_columns_2
to_predict = pd.DataFrame(columns = all_columns)

to_predict.loc[0, "car_brand"] = test_car.car_brand
to_predict.loc[0, "car_model"] = test_car.car_model
to_predict.loc[0, "car_type"] = test_car.car_type
to_predict.loc[0, "condition"] = test_car.condition
to_predict.loc[0, "engine_power"] = test_car.engine_power
to_predict.loc[0, "ccm"] = test_car.ccm
to_predict.loc[0, "service_book"] = test_car.service_book
to_predict.loc[0, "year"] = test_car.year
to_predict.loc[0, "milage"] = test_car.milage
to_predict.loc[0, "fuell"] = test_car.fuell
to_predict.loc[0, "n_people"] = test_car.n_ppl
to_predict.loc[0, "n_doors"] = test_car.n_doors
to_predict.loc[0, "airbags"] = test_car.n_airbags
to_predict.loc[0, "air_condition"] = test_car.aircondition
to_predict.loc[0, "transmission"] = test_car.transmission
to_predict.loc[0, "country_from"] = test_car.country
to_predict.loc[0, "extra_category"] = test_car.extras

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

to_predict[num_columns_2] = to_predict[num_columns_2].astype(np.float32)

#adding car category depending on brand
to_predict["ccm"] = to_predict["ccm"].astype(np.float32)
for index, row in tqdm(to_predict.iterrows()):
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
    
# num_columns = num_columns + ['pcm', 'drs_ppl', 'air_ppl', 'y_m']
# to_predict['pcm'] = to_predict.engine_power / to_predict.ccm
to_predict['y_m'] = to_predict.milage / to_predict.year
to_predict['drs_ppl'] = to_predict.n_doors / to_predict.n_people
# to_predict['air_ppl'] = to_predict.n_people / to_predict.airbags 
#-------------
#-------------END
#-------------

X_train_final = full_pipeline.fit_transform(X_train)
X_test_final = full_pipeline.transform(X_test)

def test_result(model, n_tests):
    text_file = open("learning_history.txt","a") 
    today = datetime.now()
    sum_errors = []
    prediction_all = model.predict(X_test_final)
    nn_mse = mse(y_test, prediction_all)
    nn_rmse = np.sqrt(nn_mse)
    score = model.score(X_test_final, y_test)

    if handle_skewed:
      nn_rmse = np.expm1(nn_rmse)

    for sample in range(n_tests):
        prediction = model.predict(X_test_final)[sample]
        y_real_value = y_test.iloc[sample]
        if not create_features:
          milage = X_test.iloc[sample]['milage']
          engine_power = X_test.iloc[sample]['engine_power']
          year = X_test.iloc[sample]['year']
        else:
          country = X_test.iloc[sample]['country_from']
          trans = X_test.iloc[sample]['transmission']
          fuell = X_test.iloc[sample]['fuell']
          milage = X_test.iloc[sample]['milage']
          engine_power = X_test.iloc[sample]['engine_power']
          year = X_test.iloc[sample]['year']

        brand = X_test.iloc[sample]['car_brand']
        car_model = X_test.iloc[sample]['car_model']

        if handle_skewed:
          prediction = np.expm1(prediction)
          y_real_value = np.expm1(y_real_value)
          milage = np.expm1(milage)
          engine_power = np.expm1(engine_power)
          year = np.expm1(year)

        error_percentage = ((-(y_real_value - prediction)/y_real_value) * 100)
        sum_errors.append(np.absolute(error_percentage))
        max_error = max(sum_errors)
        if not create_features:
          print("prediction: {:7.0f},  real price: {:7.0f},  percent error: {:6.2f}%, milage: {:7.0f}, power: {:4.0f}, year: {:4.0f}".format(prediction, y_real_value, error_percentage, milage, engine_power, year))
        else:
          print("prediction: {:7.0f},  real price: {:7.0f},  percent error: {:6.2f}%, country: {:20}, trans: {:20}, fuell: {:10}, brand: {:15}, model: {}".format(prediction, y_real_value, error_percentage, country, trans, fuell, brand, car_model))

    final_log = 'average error: {:7.2f}%, median error: {:7.2f}%, absolute error: {:7.0f}, score: {:7.3f}, max error: {:7.2f}%, set size: {}'.format(np.mean(sum_errors), np.median(sum_errors), nn_rmse, score, max_error, data_frame_training_ready_no_nan.shape[0])
    print(final_log)

    text_file.write("\n{}  {}".format(today, final_log))


clf = MLPRegressor(solver='adam', alpha=0.01,  
                    hidden_layer_sizes=(5, 800), random_state=42, 
                    batch_size= 32, verbose = True, max_iter = 500,
                    learning_rate = 'adaptive', warm_start=True,
                    validation_fraction = 0.1, early_stopping = True)

clf.fit(X_train_final, y_train.values)

test_result(clf, 300)

to_predict = to_predict[all_columns]

prediction_test = full_pipeline.transform(to_predict)

prediction_result = clf.predict(prediction_test)
print("\n\n", prediction_test)
print("\n\n", prediction_result)


# class CustomCallbacs():
#   earlyStop = keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True, monitor='val_loss')

# cbs = CustomCallbacs()

# def build_model(n_hidden = 3, learning_rate = 0.0001, batch_norm = True, dropout = 0.3):
#   model = keras.models.Sequential()
#   for n in range(n_hidden):
#       model.add(keras.layers.Dense(1000, kernel_initializer='uniform')) #best
#       model.add(keras.layers.ReLU()) #best
#       model.add(keras.layers.Dropout(dropout))
#   # model.add(keras.layers.Dense(100, kernel_initializer='uniform'))
#   model.add(keras.layers.Dense(1))
#   optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
#   model.compile(loss = 'mean_absolute_percentage_error', optimizer = optimizer)
#   return model

# keras_reg = build_model()
# keras_reg.fit(X_train_final, 
#                     y_train.values, 
#                     epochs = 300, 
#                     validation_split = 0.1, 
#                     callbacks = [cbs.earlyStop],
#                     batch_size = 32
#                    )

# test_result(keras_reg, 300)


# saving model
joblib.dump(clf, "final_model_v1.gz")

# saving scaler
joblib.dump(full_pipeline, "final_transofrmator_v1.gz")

