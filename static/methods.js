function checkParametres() {
	var param_list = []

	car_model = select_model_list.options[select_model_list.selectedIndex].text
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

	if (param_list.includes("výrobce") == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes("model") == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes("karoserie") == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes("převodovka") == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes("palivo") == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes(undefined) == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes('')) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.length == 0) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.includes("klimatizace") == true) {
		alert("vyplňte všechny údaje")
		return false
	} else if (param_list.length < 13) {
		alert("vyplňte všechny údaje")
		return false
	} else if (year < 2000) {
		alert("rok výroby vozidla musí být mladší 2000")
		return false
	} else if (year > 2021) {
		alert("rok výroby vozidla nesmí být v budoucnosti")
		return false
	} else {
		return true
	}
}

function passParametres() {
	var param_list = []

	car_model = select_model_list.options[select_model_list.selectedIndex].text
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
	param_list.push(country);

	console.log(param_list.slice(0, 6));
	console.log(param_list.slice(7, 12));
	return param_list
}

function clear_list() {
	select_model_list.style.backgroundColor = "white"
	var i, L = select_model_list.options.length - 1;

	for(i = L; i >= 0; i--) {
		if (select_model_list.options[i].text != "model") {
			select_model_list.remove(i);
		}
	   }
}

function display_car_models(car_brand) {
	var car_model_Object = car_brand_model_dict[car_brand];
	for(index in car_model_Object) {
			select_model_list.options[select_model_list.options.length] = new Option(car_model_Object[index], index);
	}
}

function add_listeners_color(elementID) {
	var element = document.getElementById(elementID);
	element.addEventListener("change", function() {
		element.style.backgroundColor = selected_element_color;
	})
}

function load_data_to_list(elementID, data, sort) {
	var list = document.getElementById(elementID);
	for(var i = 0; i < data.length; i++) {
		if (sort == true) {
			var opt = data.sort()[i];
			var el = document.createElement("option");
			el.text = opt;
			el.value = opt;
			list.add(el);
		} else {
			var opt = data[i];
			var el = document.createElement("option");
			el.text = opt;
			el.value = opt;
			list.add(el);
		}	
	}
}

function set_test_values() {
	document.getElementById("car_brand_list").value= "Volkswagen"
	document.getElementById("car_brand_list").text= "Volkswagen"

	document.getElementById("car_model_list").value = "Golf"
	document.getElementById("car_model_list").text = "Golf"

	document.getElementById("car_type_list").value = "hatchback"
	document.getElementById("car_type_list").text = "hatchback"

	document.getElementById("book").value = "ne"
	document.getElementById("book").text = "Ne"

	document.getElementById("condition").value = "ojeté"
	document.getElementById("condition").text = "Ojeté"

	document.getElementById("year").value = 2008
	document.getElementById("year").text = 2008

	document.getElementById("milage").value = 285000
	document.getElementById("milage").text = 285000

	document.getElementById("engine_power").value = 74
	document.getElementById("engine_power").text = 74

	document.getElementById("car_fuel_list").value = "nafta"
	document.getElementById("car_fuel_list").text = "nafta"

	document.getElementById("n_doors_list").value = 5
	document.getElementById("n_doors_list").text = 5

	document.getElementById("air_condition_list").value = "manuální"
	document.getElementById("air_condition_list").text = "manuální"

	document.getElementById("transmission_list").value = "manuální"
	document.getElementById("transmission_list").text = "manuální"

	document.getElementById("country_list").value = "Česká republika"
	document.getElementById("country_list").text = "Česká republika"
}