import requests
import pandas as pd
import numpy as np
from helium import *
from bs4 import BeautifulSoup as bs
import time
import os
import email_notification

def get_page_url(site_number):
    return f"https://www.sauto.cz/osobni/hledani#!category=1&condition=1&condition=2&condition=4&page={site_number}&sort=2"

def get_details_url(n_pages):
    car_detail_url_list = []
    index = 0
    for n in range(1, (n_pages+1)):
        browser = start_chrome(get_page_url(n), headless=True)
        soup = bs(browser.page_source, "html.parser")
        results = soup.find_all("a", class_= "toDetail")
        for element in results:
            url_part = element.get("href")
            url = f"https://www.sauto.cz{url_part}"
            if url not in car_detail_url_list:
                car_detail_url_list.append(url)
                index += 1

            print(f"urls added: {index}", end="\r", flush=True)

        # save to csv each 100 URLs
        if n % 500 == 0:
            df = pd.DataFrame(car_detail_url_list)
            df.to_csv("car_detail_url_list.csv", index = False)

        browser.quit()
        time.sleep(3)

    df = pd.DataFrame(car_detail_url_list, columns = ["url"])
    df.to_csv("car_detail_url_list.csv", index=False)
    print(f"finished with {index} urls saved")
    email_notification.send_email(f"got all {index} detail URLs")


def scrape_car_detail(detail_url):
    df_detail = pd.DataFrame()
    browser = start_chrome(url=detail_url, headless=True)
    soup = bs(browser.page_source, "html.parser")

    def handle_exception(item, print_=False):
        df_detail[item] = np.nan
        if print_:
            print(f"{item} nan for url: {detail_url}")

    try:
        span_brand = soup.find("span", class_="brand")
        car_brand = str(span_brand).split("brand\"> ", -1)[-1].split(" </span>", 1)[0]
        df_detail["car_brand"] = [car_brand]
    except Exception:
        handle_exception("car_brand")

    try:
        span_model = soup.find("span", class_="name")
        car_model = str(span_model).split("name\"> ", -1)[-1].split(" </span>", 1)[0]
        df_detail["car_model"] = [car_model]
    except Exception:
        handle_exception("car_model")

    try:
        span_detail = soup.find("span", {"data-sticky-headheader-value-src": "catalogue"})
        detail = str(span_detail).split("catalogue\"> ", -1)[-1].split(" </span>", 1)[0]
        df_detail["detail"] = [detail]
    except Exception:
        handle_exception("detail")

    try:
        span_price = soup.find("strong", itemprop="price")
        price = str(span_price).split("price\">", -1)[-1].split("</strong>", 1)[0].replace("\xa0", "")
        df_detail["price"] = [price]
    except Exception:
        handle_exception("price")

    try:
        span_year = soup.find("td", {"data-sticky-header-value-src": "year"})
        year = str(span_year).split("year\">", -1)[-1].split("</td>", 1)[0]
        if len(year) > 4:
            year = year.split("/")[1]

        df_detail["year"] = [year]
    except Exception:
        handle_exception("year")

    try:
        span_milage = soup.find("span", {"class": "vin_detail"})
        milage = str(span_milage).split("vin_detail\">", -1)[-1].split(" km</span>", 1)[0].replace("\xa0", "")
        df_detail["milage"] = [milage]
    except Exception:
        handle_exception("milage")

    try:
        additional_info = soup.find("div", {"id": "description"})
        additional_info = str(additional_info).split("itemprop=\"description\">", 1)[1].split("</p> </div>", 1)[0]
        df_detail["additional_info"] = [additional_info]
    except Exception:
        handle_exception("additional_info")

    try:
        extras = soup.find("div", {"id": "equipment"})
        equipment_list = str(extras).split("<li>")[1:]
        equipment = [(item.split("</li>")[0]) for item in equipment_list]
        df_detail["extras_list"] = [str(equipment)]
    except Exception:
        handle_exception("extra_list")

    #table_span
    table_span = soup.find("table", {"id":"detailParams"})

    try:
        fuell = str(table_span).split("Palivo:</th><td>", 1)[1].split("</td></tr>")[0]
        df_detail["fuell"] = [fuell]
    except Exception:
        handle_exception("fuell")

    try:
        ccm = str(table_span).split("Objem:</th><td>", 1)[1].split("</td></tr>")[0].replace("\xa0", "").replace(" ccm", "")
        df_detail["ccm"] = [ccm]
    except Exception:
        handle_exception("ccm")

    try:
        engine_power = str(table_span).split("Výkon:</th><td>", 1)[1].split(" kW", 1)[0]
        df_detail["engine_power"] = [engine_power]
    except Exception:
        handle_exception("engine_power")

    try:
        transmission = str(table_span).split("Převodovka:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["transmission"] = [transmission]
    except Exception:
        handle_exception("transmission")

    try:
        air_condition = str(table_span).split("Klimatizace:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["air_condition"] = [air_condition]
    except Exception:
        handle_exception("air_condition")

    try:
        vin = str(table_span).split("vin_detail\"> ", 1)[1].split("<", 1)[0].replace(" ", "")
        df_detail["vin"] = [vin]
    except Exception:
        handle_exception("vin")

    try:
        service_book = str(table_span).split("knížka:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["service_book"] = [service_book]
    except Exception:
        handle_exception("service_book")

    try:
        price_more_info = str(table_span).split("Poznámka k ceně:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["price_more_info"] = [price_more_info]
    except Exception:
        handle_exception("price_more_info")

    try:
        country_from = str(table_span).split("původu:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["country_from"] = [country_from]
    except Exception:
        handle_exception("country_from")

    try:
        condition = str(table_span).split("itemCondition\">", 1)[1].split("</td></tr>", 1)[0]
        df_detail["condition"] = [condition]
    except Exception:
        handle_exception("condition")

    try:
        n_doors = str(table_span).split("dveří:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["n_doors"] = [n_doors]
    except Exception:
        handle_exception("n_doors")

    try:
        n_people = str(table_span).split("Počet míst:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["n_people"] = [n_people]
    except Exception:
        handle_exception("n_people")

    try:
        car_type = str(table_span).split("Karoserie:</th><td>", 1)[1].split("</td></tr>", 1)[0]
        df_detail["car_type"] = [car_type]
    except Exception:
        handle_exception("car_type")

    try:
        add_url = detail_url.split("?", 1)[0]
        df_detail["add_id-href"] = [f"{add_url}?goFrom=list#"]
    except Exception:
        handle_exception("add_id-href")

    browser.quit()
    time.sleep(3)

    return df_detail


def run_scrapping():
    # run through all pages and get all the URLs of add details, save to CSV
    get_details_url(400)

    directory = "car_update_data"
    csv_file_name = "car_list_all_v2_sauto_update.csv"

    if os.path.exists(f"{directory}/{csv_file_name}"):
        car_list_all_v2_sauto_update = pd.read_csv(f"{directory}/{csv_file_name}")
        print("loading existing dataframe")
    else:
        os.makedirs(directory)
        car_list_all_v2_sauto_update = pd.DataFrame(columns=["car_brand", "car_model"])
        print("creating new dataframe")

    car_detail_url_list = pd.read_csv("car_detail_url_list.csv")

    for index, url in car_detail_url_list.iterrows():
        detail_df = scrape_car_detail(url[0])
        car_list_all_v2_sauto_update = car_list_all_v2_sauto_update.append(detail_df, sort=False)
        print(f"details added: {index + 1}", end="\r", flush=True)
        if (index + 1) % 300 == 0:
            car_list_all_v2_sauto_update.to_csv(f"{directory}/{csv_file_name}", index=False)
            print("300 more details added, saving...")

    car_list_all_v2_sauto_update.to_csv(f"{directory}/{csv_file_name}", index=False)
    email_notification.send_email(f"got all the {index + 1} details")
    print(car_list_all_v2_sauto_update.head())


run_scrapping()