import pandas as pd
import numpy as np

data_frame_training_ready = pd.read_csv('/Users/jankolnik/Downloads/car_list_sauto_preprocessed_2.csv')

def get_car(prediction):
  raw_data = data_frame_training_ready.copy()
  result = [x.strip() for x in prediction.split(',')]
  if "Česká republika" not in prediction:
    country = [x.strip() for x in result[3].split(' ')][-1]
  else:
    country = "Česká republika"
  price = int([x.strip() for x in result[1].split(' ')][-1])
  
  for brand in ['Alfa Romeo', "Great Wall", 'Aston Martin', "Land Rover"]:
    if brand in result[6]:
      car_brand = brand
    else:
      car_brand = [x.strip() for x in result[6].split(' ')][1]

  if len([x.strip() for x in result[7].split(' ')]) == 3:
    model_1 = [x.strip() for x in result[7].split(' ')][-2]
    model_2 = [x.strip() for x in result[7].split(' ')][-1]
    car_model = model_1 + " " + model_2
  else:
    car_model = [x.strip() for x in result[7].split(' ')][-1]

  year = float([x.strip() for x in result[8].split(' ')][-1]) + 2000
  milage = np.float64([x.strip() for x in result[9].split(' ')][-1])
  engine_power = int([x.strip() for x in result[10].split(' ')][-1])

  filtered_df = raw_data[(raw_data["car_brand"]==car_brand)
   & (raw_data["car_model"]==car_model)
     & (raw_data["price"]==price)
       & (raw_data["year"]==year)
        & (raw_data["milage"]==milage)
         & (raw_data["engine_power"]==engine_power)
         ]
  pd.options.display.max_colwidth = 500
  for index, row in filtered_df.iterrows():
    print(row[["car_model", "year", "milage", "engine_power", "price", "detail", "additional_info", "add_id-href"]])

prediction_text = input("paste prediction text: ")
get_car(prediction_text)