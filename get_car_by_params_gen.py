import csv   

def get_car(prediction, data_source):
    #   raw_data = data_frame_training_ready.copy()
    result = [x.strip() for x in prediction.split(',')]
    if "Česká republika" not in prediction:
        country = [x.strip() for x in result[3].split(' ')][-1]
    else:
        country = "Česká republika"
    price = float([x.strip() for x in result[1].split(' ')][-1])

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
    milage = float([x.strip() for x in result[9].split(' ')][-1])
    engine_power = int([x.strip() for x in result[10].split(' ')][-1])

    filtered_pipeline = (
        raw_data for raw_data in
        raw_data if
        raw_data["car_brand"]==car_brand and raw_data["car_model"]==car_model 
        and raw_data["year"]==str(year) and raw_data["milage"]==str(milage) and raw_data["price"]==str(price)
        )

    columns_to_print = ["car_model", "year", "milage", "engine_power", "price", "price_more_info", "detail", "additional_info", "add_id-href"]
    row = next(filtered_pipeline)

    [print(row[column]) for column in columns_to_print]



with open('/Users/jankolnik/Desktop/ML_projects/car_price_data/car_list_sauto_preprocessed_2.csv', mode="r") as file:
    data = csv.DictReader(file)

    raw_data = (line for line in data)

    prediction_text = input("paste prediction text: ")

    get_car(prediction_text, raw_data) 




"""
pred:  239462, real:  219000, err.rate:   9.34%, country: Česká republika , trans: manuální     , fuell: nafta   , br: Citroën        , md: Berlingo     , year: 15.0, milage:  86000, pwr: 84
pred:  695217, real:  598000, err.rate:  16.26%, country: Česká republika , trans: automatická  , fuell: nafta   , br: Volkswagen     , md: Passat       , year: 18.0, milage:  52534, pwr: 140
pred:  272216, real:  250000, err.rate:   8.89%, country: Česká republika , trans: manuální     , fuell: benzín  , br: Dacia          , md: Duster       , year: 16.0, milage:  50713, pwr: 84
pred:  388152, real:  619950, err.rate: -37.39%, country: Česká republika , trans: automatická  , fuell: benzín  , br: Hyundai        , md: Kona         , year: 19.0, milage:  12500, pwr: 77
pred:  354201, real:  469999, err.rate: -24.64%, country: Německo         , trans: manuální     , fuell: nafta   , br: Mercedes-Benz  , md: GLA          , year: 14.0, milage: 123615, pwr: 100
pred:  385986, real:  349000, err.rate:  10.60%, country: Belgie          , trans: manuální     , fuell: nafta   , br: Citroën        , md: Grand C4 Picasso, year: 16.0, milage:  85581, pwr: 110
"""