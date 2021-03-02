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
        print("pred: {:7.0f}, real: {:7.0f}, err.rate: {:6.2f}%, country: {:16}, trans: {:13}, fuell: {:8}, br: {:15}, md: {:13}, year: {:4}, milage: {:6.0f}, pwr: {:.0f}".format(prediction, y_real_value, error_percentage, country, trans, fuell, brand, car_model, year, milage, engine_power))

    final_log = 'average error: {:7.2f}%, median error: {:7.2f}%, absolute error: {:7.0f}, score: {:7.3f}, max error: {:7.2f}%, set size: {}'.format(np.mean(sum_errors), np.median(sum_errors), nn_rmse, score, max_error, data_frame_training_ready_no_nan.shape[0])
    print(final_log)

    text_file.write("\n{}  {}".format(today, final_log))