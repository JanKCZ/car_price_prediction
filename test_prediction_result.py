from sklearn.metrics import mean_squared_error as mse
import numpy as np
import pandas as pd

def test_result(model, n_tests, test_data, test_labels, library="scikit"):
    """
    params: library - default "scikit", other option: "torch"
    params: test_data - for "torch" library, input data in torch tensor
    params: test_labels - for "torch" library, input data in torch tensor
    """
    text_file = open("learning_history.txt","a")
    today = datetime.datetime.now()
    sum_errors = []
    if library == "scikit":
        prediction_all = model.predict(test_data)
        nn_mse = mse(test_labels, prediction_all)
        nn_rmse = np.sqrt(nn_mse)
        score = model.score(test_data, test_labels)
    elif library == "torch":
        prediction_all = model(test_data.float())
        nn_mse = mse(test_labels.numpy(), prediction_all.detach().numpy())
        nn_rmse = np.sqrt(nn_mse)
        score = 0
    else:
        print("error, choose scikit or torch library")

    for sample in range(n_tests):
        if library == "scikit":
            prediction = model.predict(test_data)[sample]
            y_real_value = test_labels.iloc[sample]
            country = X_test.iloc[sample]['country_from']
            trans = X_test.iloc[sample]['transmission']
            fuell = X_test.iloc[sample]['fuell']
            milage = X_test.iloc[sample]['milage']
            engine_power = X_test.iloc[sample]['engine_power']
            year = X_test.iloc[sample]['year']
            brand = X_test.iloc[sample]['car_brand']
            car_model = X_test.iloc[sample]['car_model']
        else:
            prediction = model(test_data[sample].float()).item()
            y_real_value = test_labels[sample].item()
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
        print("pred: {:7.0f}, real: {:7.0f}, err.rate: {:6.2f}%, country: {:16}, trans: {:13}, fuell: {:8}, br: {:15}, md: {:13}, year: {:4}, milage: {:6.0f}, pwr: {:.0f}".format(prediction, y_real_value, error_percentage, country, trans, fuell, brand, car_model, year, milage, engine_power))

    final_log = 'average error: {:7.2f}%, median error: {:7.2f}%, absolute error: {:7.0f}, score: {:7.3f}, max error: {:7.2f}%, set size: {}'.format(np.mean(sum_errors), np.median(sum_errors), nn_rmse, score, max_error, data_frame_training_ready_no_nan.shape[0])
    print(final_log)

    text_file.write("\n{}  {}".format(today, final_log))