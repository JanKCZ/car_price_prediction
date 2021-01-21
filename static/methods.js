function passParametres() {
	var param_list = []

	car_model = select_model_list.options[select_model_list.selectedIndex].text
	// car_model = select_model_list.text
	param_list.push(car_brand)
	document.getElementById("car_brand_input").value = car_brand
	param_list.push(car_model)
	document.getElementById("car_model_input").value = car_model

	var type_list = document.getElementById("car_type_list")
	var type = type_list.options[type_list.selectedIndex].text
	document.getElementById("car_type_input").value = type
	param_list.push(type)

	var book_list = document.getElementById("book")
	var book = book_list.options[book_list.selectedIndex].value
	document.getElementById("service_book_input").value = book
	param_list.push(book)

	var condition_list = document.getElementById("condition")
	var condition = condition_list.options[condition_list.selectedIndex].value
	document.getElementById("condition_input").value = condition
	param_list.push(condition)

	var year = document.getElementById("year").value
	document.getElementById("year_input").value = year
	param_list.push(year)

	var milage = document.getElementById("milage").value
	document.getElementById("milage_input").value = milage
	param_list.push(milage)

	var engine_power = document.getElementById("engine_power").value
	document.getElementById("engine_power_input").value = engine_power
	param_list.push(engine_power)

	var fuel_list = document.getElementById("car_fuel_list")
	var fuel = fuel_list.options[fuel_list.selectedIndex].text
	document.getElementById("fuell_input").value = fuel
	param_list.push(fuel)

	var n_ppl = document.getElementById("n_ppl_list").value
	document.getElementById("n_ppl_input").value = n_ppl
	param_list.push(n_ppl)

	var airbags = document.getElementById("n_airbags_list").value
	document.getElementById("n_airbags_input").value = airbags
	param_list.push(airbags)

	var doors = document.getElementById("n_doors_list").value
	document.getElementById("n_doors_input").value = doors
	param_list.push(doors)

	var aircondition_list = document.getElementById("air_condition_list")
	var aircondition = aircondition_list.options[aircondition_list.selectedIndex].text
	document.getElementById("aircondition_input").value = aircondition
	param_list.push(aircondition)

	var transmission_list = document.getElementById("transmission_list")
	var transmission = transmission_list.options[transmission_list.selectedIndex].text
	document.getElementById("transmission_input").value = transmission
	param_list.push(transmission)

	var country_list = document.getElementById("country_list")
	var country = country_list.options[country_list.selectedIndex].text
	document.getElementById("country_input").value = country
	param_list.push(country)

	var extras_list = document.getElementById("extras_list")
	var extras = extras_list.options[extras_list.selectedIndex].value
	document.getElementById("extras_input").value = extras
	param_list.push(extras)

	console.log(param_list)

	if (param_list.includes("-- vyberte jednu z možností --") == true) {
		alert("vyplňte všechny údaje")
	} else if (param_list.includes(undefined)) {
		alert("vyplňte všechny údaje")
	} else if (param_list.includes("")) {
		alert("vyplňte všechny údaje")
	} else {
		alert("brand: " + car_brand + "\n" +
			"model: " + car_model + "\n" +
			"type: " + type + "\n" +
			"engine_power: " + engine_power + "\n" +
			"condition: " + condition + "\n" +
			"book: " + book + "\n" +
			"year: " + year + "\n" +
			"milage: " + milage + "\n" +
			"fuel: " + fuel + "\n" +
			"n_ppl: " + n_ppl + "\n" +
			"airbags: " + airbags + "\n" +
			"doors: " + doors + "\n" +
			"aircondition: " + aircondition + "\n" +
			"transmission: " + transmission + "\n" +
			"country: " + country + "\n" +
			"extras: " + extras);
	}
}