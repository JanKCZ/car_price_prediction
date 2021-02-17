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
from sklearn.ensemble import BaggingRegressor
import sklearn
import math
from sklearn.neural_network import MLPRegressor
# from tensorflow.keras import layers
# import tensorflow as tf
from tqdm import tqdm
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

def yes_no(answer):
  user_input = input(answer)
  while True:
    if user_input == "y":
      return True
    elif user_input == "n":
      return False
    else:
      print("answer y or n")
      break
    
if yes_no("update model? y/n: "):
  print("loading CSV file....")
  raw_data = pd.read_csv('/Users/jankolnik/Downloads/car_list_all_v1_updated_sauto.csv')

  raw_data_update = pd.read_csv("/Users/jankolnik/Downloads/car_list_all_v1_sauto_update.csv")

  raw_data_updated = pd.concat([raw_data, raw_data_update])

  #drop duplicated adds
  raw_data_updated = raw_data_updated.drop_duplicates(subset=['add_id-href'])
  raw_data = raw_data.reset_index(drop=True)

  #save to CSV
  raw_data_updated.to_csv('/Users/jankolnik/Downloads/car_list_all_v1_updated_sauto.csv', index = False, header=True)

  print("{} shape before update".format(raw_data.shape[0]))
  print("added {} rows of raw data".format(raw_data_update.shape[0]))
  print("final raw data shape is {}".format(raw_data_updated.shape))

  #remove adds, which include words about damaged or non-functional cars
  bad_words = [" vadný", " vadné", " rozbit", " havarovan", " poškozen", " špatn", "nepojízd", " bourané"]
  good_words = ["bez poškození", "žádné poškození", "nemá poškození", "není poškozen"]
  bad_index = []

  not_nan = raw_data[raw_data.additional_info.notnull()]

  for word in bad_words:
    bad_words_index = not_nan[not_nan.additional_info.str.contains(word, case = False)].index
    for index in bad_words_index:
      if index not in bad_index:
        bad_index.append(index)

  for word in bad_words:
    bad_words_index = raw_data[raw_data.detail.str.contains(word, case = False)].index
    for index in bad_words_index:
      if index not in bad_index:
        bad_index.append(index)

  raw_data_updated = raw_data_updated.drop(bad_index)

  pd.set_option('display.max_colwidth', None)
  print("dropping wrong adds with words: ", bad_words)

  #dropping useless columns:
  data_no_trash_columns = raw_data_updated.drop(['web-scraper-order', 'web-scraper-start-url', 
                                                'detail-href', 'detail', 
                                                'in_usage_from_year', 'additional_info', 
                                                'price_info', 'gps', 'gps-href', 'warranty_until', 'euro_certification',
                                                'first_owner', 'vin', 'add_id', 'add_id-href', 'seller_name'], axis = 1)

  data_frame_extras = data_no_trash_columns.copy()

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
  data_price_less_1k = data_frame_no_dupl[lambda data: data.price <= 5000].index
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

  data_frame_training_ready = data_frame_no_neuvedeno.copy()

  print("saving to CSV, preprocessed, shape: ", data_frame_training_ready.shape)
  data_frame_training_ready.to_csv ('/Users/jankolnik/Downloads/car_list_sauto_preprocessed_2.csv', index = False, header=True)

drop_nan = True               # use KNN imputer to impute num and cat values
drop_year = 2000
is_sparse = False
drop_wrong_power_ccm_combination = True

data_frame_training_ready = pd.read_csv('/Users/jankolnik/Downloads/car_list_sauto_preprocessed_2.csv')
print("size of fresh CSV: ", data_frame_training_ready)

# drop old cars with unreliable data
data_older_cars = data_frame_training_ready[lambda data: data.year < drop_year].index
data_frame_training_ready = data_frame_training_ready.drop(data_older_cars)
print("dropping older cars than: ", drop_year)

print("dropping ostatní brands")
data_frame_training_ready = data_frame_training_ready.drop(data_frame_training_ready[lambda data: data.car_brand == 'Ostatní'].index)

#change milage to 1, so years can be divided by it
data_frame_training_ready.loc[data_frame_training_ready['milage'] == 0, 'milage'] = 1
#years will only be fro 1 to 11
data_frame_training_ready.loc[data_frame_training_ready['year'] != np.nan, 'year'] = data_frame_training_ready['year'] - 2000

#dropping older cars with way to low milage
older_cars = data_frame_training_ready[lambda data: data.year < 8].index
old_and_low_milage = data_frame_training_ready.loc[older_cars][lambda data: data.milage < 600].index
data_frame_training_ready = data_frame_training_ready.drop(old_and_low_milage)

print("using ordinal on service_book, air_condition, condition")
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

data_frame_training_ready.transmission = data_frame_training_ready.transmission.replace(['manuální (6 stupňová)', 'manuální (5 stupňová)', 'manuální (8 stupňová a více)', 'manuální (7 stupňová)', 'manuální (4 stupňová)', 'manuální (3 stupňová a méně)'], 'manuální')
data_frame_training_ready.transmission = data_frame_training_ready.transmission.replace(['poloautomatická (8 stupňová a více)', 'poloautomatická (5 stupňová)', 'poloautomatická', 'poloautomatická (6 stupňová)', 'poloautomatická (7 stupňová)'], 'poloautomatická')
data_frame_training_ready.transmission = data_frame_training_ready.transmission.replace(['automatická (6 stupňová)', 'automatická (8 stupňová a více)','automatická (7 stupňová)', 'automatická (4 stupňová)', 'automatická (5 stupňová)', 'automatická (3 stupňová a méně)'], 'automatická')

cat_columns = ['fuell','transmission', 'car_model', 'car_brand', 'country_from', 'car_type']
num_columns = ['engine_power', 'year', 'milage', 'service_book', 'condition', 'air_condition', 'n_doors']

# for cars with electricity, set value of ccm = 0
data_frame_training_ready.loc[data_frame_training_ready['fuell'] == "elektro", 'ccm'] = 0

#drop_wrong_power_ccm_combination:
small_ccm_smaller_580 = data_frame_training_ready[lambda data: data.ccm < 580]
small_ccm_bigger_0 = small_ccm_smaller_580[lambda data: data.ccm > 0]
small_power_to_ccm = small_ccm_bigger_0[lambda data: data.engine_power > 10].index
data_frame_training_ready = data_frame_training_ready.drop(small_power_to_ccm)

big_ccm = data_frame_training_ready[lambda data: data.ccm < 3500]
big_power_to_ccm = big_ccm[lambda data: data.engine_power > 450].index
data_frame_training_ready = data_frame_training_ready.drop(big_power_to_ccm)

cmm_1000 = data_frame_training_ready[lambda data: data.ccm > 1000]
ccm_1000_power_20 = cmm_1000[lambda data: data.engine_power < 20].index
data_frame_training_ready = data_frame_training_ready.drop(ccm_1000_power_20)

important_columns = num_columns + cat_columns

print("size before NAN drop: ", data_frame_training_ready.shape)
data_frame_training_ready_no_nan = data_frame_training_ready[important_columns + ['price']].dropna()
print("size after dropping NAN: ", data_frame_training_ready_no_nan.shape)

print("size before droping inf: ", data_frame_training_ready_no_nan.shape)
data_frame_training_ready_no_nan = data_frame_training_ready_no_nan.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
print("size after droping inf", data_frame_training_ready_no_nan.shape)

data_frame_training_ready_no_nan[cat_columns] = data_frame_training_ready_no_nan[cat_columns].astype(np.str)

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

X_train_final = full_pipeline.fit_transform(X_train)
X_test_final = full_pipeline.transform(X_test)

def test_result(model, n_tests):
    create_features = False
    text_file = open("learning_history.txt","a") 
    today = datetime.now()
    sum_errors = []
    prediction_all = model.predict(X_test_final)
    nn_mse = mse(y_test, prediction_all)
    nn_rmse = np.sqrt(nn_mse)
    score = model.score(X_test_final, y_test)

    for sample in range(n_tests):
        prediction = model.predict(X_test_final)[sample]
        y_real_value = y_test.iloc[sample]
        country = X_test.iloc[sample]['country_from']
        trans = X_test.iloc[sample]['transmission']
        fuell = X_test.iloc[sample]['fuell']
        milage = X_test.iloc[sample]['milage']
        engine_power = X_test.iloc[sample]['engine_power']
        year = X_test.iloc[sample]['year']

        brand = X_test.iloc[sample]['car_brand']
        car_model = X_test.iloc[sample]['car_model']

        error_percentage = ((-(y_real_value - prediction)/y_real_value) * 100)
        sum_errors.append(np.absolute(error_percentage))
        max_error = max(sum_errors)
        print("pred: {:7.0f}, real: {:7.0f}, err.rate: {:6.2f}%, country: {:16}, trans: {:13}, fuell: {:8}, model: {:15} {:13}, year: {:4}, milage: {:6.0f}, pwr: {0f}".format(prediction, y_real_value, error_percentage, country, trans, fuell, brand, car_model, year, milage, engine_power))

    final_log = 'average error: {:7.2f}%, median error: {:7.2f}%, absolute error: {:7.0f}, score: {:7.3f}, max error: {:7.2f}%, set size: {}'.format(np.mean(sum_errors), np.median(sum_errors), nn_rmse, score, max_error, data_frame_training_ready_no_nan.shape[0])
    print(final_log)

    text_file.write("\n{}  {}".format(today, final_log))


clf = MLPRegressor(solver='adam', alpha=0.001, learning_rate_init=0.001,
                    hidden_layer_sizes=(20, 400), random_state=42,
                    batch_size= 32, verbose = True, max_iter = 500,
                    learning_rate = 'adaptive', warm_start=True,
                    validation_fraction = 0.1, early_stopping = True)


clf.fit(X_train_final, y_train.values)

test_result(clf, 300)




# def load_model():
# 	model_path = "/Users/jankolnik/Downloads/final_model_v1.gz"
# 	return joblib.load(model_path)
# model = load_model()
# test_result(model, 300)

# class CustomCallbacs():
#   earlyStop = tf.keras.callbacks.EarlyStopping(patience = 5, restore_best_weights = True, monitor='val_loss')

# cbs = CustomCallbacs()

# def build_model(n_hidden = 3, learning_rate = 0.0001, batch_norm = True, dropout = 0.1):
#   model = tf.keras.models.Sequential()
#   l1 = tf.keras.regularizers.l1

#   for n in range(n_hidden):
#       model.add(tf.keras.layers.Dense(1000, kernel_initializer='uniform', kernel_regularizer=l1(0.0001))) #best
#       model.add(tf.keras.layers.ReLU()) #best
#       model.add(tf.keras.layers.Dropout(dropout))
    
#   model.add(tf.keras.layers.Dense(10, kernel_initializer='uniform'))
#   #model.add(keras.layers.Dropout(dropout))
#   model.add(tf.keras.layers.Dense(1))

#   optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
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

