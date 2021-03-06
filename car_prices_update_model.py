#!/usr/bin/env python
# coding: utf-8

# # Purpose of this file
# ### process new data from .csv file and output updated model, with predict and score methods

# MODULE IMPORT
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split

import joblib
from tqdm import tqdm
import re

# FILE IMPORT
from car_prediction_pytorch_model import *
from test_prediction_result import *
from car_prediction_scikit_model import *

data_source_path = "/Users/jankolnik/Desktop/ML_projects/car_price_data/"

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
    
default = False
# yes_no("update model? [y/n]: ")

if yes_no("update model? [y/n]: "):
  print("loading CSV file....")
  raw_data = pd.read_csv(f'{data_source_path}car_list_all_v1_updated_sauto.csv')

  raw_data_update = pd.read_csv(f"{data_source_path}car_list_all_v2_sauto_update.csv")

  raw_data_updated = pd.concat([raw_data, raw_data_update])
  
  #drop duplicated adds
  raw_data_updated.drop_duplicates(subset=['add_id-href'], keep = "last", inplace = True)
  
  raw_data_updated = raw_data_updated.reset_index(drop=True)
  # raw_data = raw_data.reset_index(drop=True)

  #save to CSV
  raw_data_updated.to_csv(f'{data_source_path}car_list_all_v1_updated_sauto.csv', index = False, header=True)

  print("{} shape before update".format(raw_data.shape[0]))
  # print("added {} rows of raw data".format(raw_data_update.shape[0]))
  print("final raw data shape is {}".format(raw_data_updated.shape))

  #remove adds, which include words about damaged or non-functional cars
  bad_words = [" vada", "vadný", "vadny", "vadné", "vadne", "vadná", " rozbit", " havarovan", " poškozen", " poskozen", "špatn", "nepojízd", "nepojizdn", 
  " bourané", " bourane", " bouraný", " bourany", "koroze", "kosmetick", "dodělaní", "na náhradní díly", "na nahradni dily", "porucha", " porouchan", " KO!",
  "drobné závady", "zavady", "závad", "oděrky", "zreziv", "rezav", "přetržený", "pretrzeny", "praskl", "nenastartuje", "nenaskočí", "problém s", "netopi", "netopí", "nejede",
  "zreziv", " vada", "po výměmě motoru", "odřeniny", "promacknut", "promáčknut", "neřadí"]
  good_words = ["bez poškození", "žádné poškození", "nemá poškození", "není poškozen", "bez koroze", 
  "žádné závady", "bez závad"]

  def clean_bad_words():
      def replace(old, new, full_text):
        return re.sub(re.escape(old), new, full_text, flags=re.IGNORECASE)

      indexes_to_remove = []

      for cat in tqdm(["price_more_info", "additional_info", "detail"]):
          not_nan = raw_data_updated[raw_data_updated[cat].notnull()]

          for good in good_words:
              not_nan[cat] = not_nan[cat].apply(lambda x: replace(good, "", x))
          
          for bad in bad_words:
              bad_words_idx = not_nan[not_nan[cat].str.contains(bad, case = False)].index

              for i in bad_words_idx:
                  if i not in indexes_to_remove:
                      indexes_to_remove.append(i)

      return indexes_to_remove
            
  index_to_drop = clean_bad_words()
  raw_data_updated = raw_data_updated.drop(index=index_to_drop, axis=0)

  pd.set_option('display.max_colwidth', None)
  print("dropped adds with words: ", bad_words)

  #dropping useless columns:
  data_no_trash_columns = raw_data_updated.drop(['web-scraper-order', 'web-scraper-start-url', 
                                                'detail-href', 
                                                'in_usage_from_year', 
                                                'price_info', 'gps', 'gps-href', 'warranty_until', 'euro_certification',
                                                'first_owner', 'vin', 'add_id', 'seller_name'], axis = 1)

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

  data_frame_number_clean.milage = pd.to_numeric(data_frame_number_clean.milage, errors='coerce')
  data_frame_number_clean.price = pd.to_numeric(data_frame_number_clean.price, errors='coerce')
  data_frame_number_clean.engine_power = pd.to_numeric(data_frame_number_clean.engine_power, errors='coerce')
  data_frame_number_clean.year = pd.to_numeric(data_frame_number_clean.year, errors='coerce')
  data_frame_number_clean.ccm = pd.to_numeric(data_frame_number_clean.ccm, errors='coerce')

  data_frame_no_dupl = data_frame_number_clean.copy()

  data_frame_no_dupl = data_frame_no_dupl.drop_duplicates(subset=['milage', 'year', 'price', 'engine_power'])
  data_frame_no_dupl = data_frame_no_dupl.replace([np.inf, -np.inf], np.nan)

  print("dropping cars for less than 10000 Kc and oveer 5 mil Kc")
  data_price_more_5m = data_frame_no_dupl[lambda data: data.price > 5000000].index
  data_price_less_1k = data_frame_no_dupl[lambda data: data.price <= 10000].index
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

  print("droping None cars")
  data_frame_no_dupl_no_none_cars = data_frame_no_dupl[lambda data: data.car_brand == "None"].index
  data_frame_no_dupl = data_frame_no_dupl.drop(data_frame_no_dupl_no_none_cars)

  data_frame_no_neuvedeno = data_frame_no_dupl.copy()
  for column in data_frame_no_neuvedeno.columns:
      data_frame_no_neuvedeno.loc[(data_frame_no_neuvedeno[column] == 'neuvedeno'), column] = np.nan

  data_frame_training_ready = data_frame_no_neuvedeno.copy()

  print("saving to CSV, preprocessed, shape: ", data_frame_training_ready.shape)
  data_frame_training_ready.to_csv (f'{data_source_path}car_list_sauto_preprocessed_2.csv', index = False, header=True)

drop_nan = True               # use KNN imputer to impute num and cat values
drop_year = 2000
is_sparse = False
drop_wrong_power_ccm_combination = True

data_frame_training_ready = pd.read_csv(f'{data_source_path}car_list_sauto_preprocessed_2.csv')

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

important_columns = num_columns + cat_columns + ["price"]
data_frame_training_ready_important = data_frame_training_ready[important_columns]

print("size before NAN drop: ", data_frame_training_ready_important.shape)
# data_frame_training_ready_important.replace([np.inf, -np.inf], np.nan, inplace=True)
data_frame_training_ready_no_nan = data_frame_training_ready_important.dropna()
print("size after dropping NAN: ", data_frame_training_ready_no_nan.shape)

data_frame_training_ready_no_nan[cat_columns] = data_frame_training_ready_no_nan[cat_columns].astype(np.str)

y = data_frame_training_ready_no_nan.pop('price').astype(np.float64)
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

X_train_final = X_train_final.astype(np.float64)
X_test_final = X_test_final.astype(np.float64)

use_scikit = True

if use_scikit:
  # SCIKIT LEARN DNN
  model = MLPRegressor(solver="adam", alpha=0.001, learning_rate_init=0.005,
                        hidden_layer_sizes=(16, 536), random_state=42,
                        batch_size= 32, verbose = True, max_iter = 300,
                        learning_rate = 'adaptive', warm_start=False,
                        validation_fraction = 0.1, early_stopping = True, activation="relu")
  
  model.fit(X_train_final, y_train.values)

  test_result(model, 300, X_test_final, y_test, X_test)
else:
  # PYTORCH
  model = Torch_model(X_train_final.shape[1], 200, 20, 0, solver=None, activation=None)

  X_train_final_torch, y_train_torch, X_test_final_torch, y_test_torch = prepare_data(X_train_final = X_train_final, 
                                                                                      y_train = y_train, 
                                                                                      X_test_final = X_test_final, 
                                                                                      y_test = y_test)
  
  trained_model = train_model(X_train_final_torch, X_test_final_torch, model, epochs=200, learning_rate=0.001)
  
  X_test_test  = torch.tensor(X_test_final)
  test_result(trained_model, 300, X_test_test, y_test_torch, X_test, library="torch")
    

# saving model
joblib.dump(model, "final_model_v1.gz")

# saving scaler
joblib.dump(full_pipeline, "final_transofrmator_v1.gz")