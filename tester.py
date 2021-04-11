import pandas as pd
import numpy as np
from time import time
import multiprocessing as mp



raw_data_updated = pd.read_csv('/Users/jankolnik/Downloads/car_list_all_v2_sauto_update.csv', dtype=str)


#remove adds, which include words about damaged or non-functional cars
bad_words = [" vadný", " vadny", " vadné", " vadne", " rozbit", " havarovan", " poškozen", " poskozen", "špatn", "nepojízd", "nepojizdn", 
" bourané", " bourane", " bouraný", " bourany", "koroze", "kosmetick", "dodělaní", "na náhradní díly", "na nahradni dily", "porucha", " porouchan", " KO!",
"drobné závady", "zavady", "závad", "oděrky", "zreziv", "rezav", "přetržený", "pretrzeny", "praskl", "nenastartuje", "nenaskočí", "problém s", "netopi", "netopí", "nejede",
"zreziv", " vada", "financování"]
good_words = ["bez poškození", "žádné poškození", "nemá poškození", "není poškozen", "bez koroze", 
"žádné závady", "bez závad", "financování"]
bad_index = []

def remove_bad_words():
    not_nan = raw_data_updated[raw_data_updated.price_more_info.notnull()]

    for word in bad_words:
        bad_words_index = not_nan[not_nan.price_more_info.str.contains(word, case = False)].index
    for good in good_words:
        if good not in bad_words_index:
            for index in bad_words_index:
                
                if index not in bad_index:
                    bad_index.append(index)
    print(len(bad_index))

def remove_bad_words_comprehention():
    # bad_index_map = [row if bad_words not in row["price_more_info"] for row in raw_data_updated (if raw_data_updated["price_more_info"].notnull() == True)]
    # bad_index_map = [row if row["price_more_info"].notnull() else False for row in raw_data_updated]
    for bad_word, good_word in zip(bad_words, good_words):
        bad_index_map = [row if bad_word in raw_data_updated["price_more_info"] and good_word not in raw_data_updated["price_more_info"]
        else False for row in raw_data_updated["price_more_info"]]

    print(len(bad_index_map))


# remove_bad_words()
# remove_bad_words_comprehention()

def time_it(func):
    def wrapper(*args, **kwarg):
        before = time()
        data = func(*args, **kwarg)
        after = time()
        print(f"took exactly {after - before}")
        return data
    return wrapper

@time_it   
def clean_bad_words():
    indexes_to_remove = []
    for cat in ["price_more_info", "additional_info", "detail"]:
        not_nan = raw_data_updated[raw_data_updated[cat].notnull()]
        for bad in bad_words:
            bad_words_index = not_nan[not_nan[cat].str.contains(bad, case = False)].index
            not_bad = not_nan.loc[bad_words_index, :].copy()
        for good in good_words:
            good_and_bad_index = not_bad[not_bad[cat].str.contains(good, case = False)].index
        for word in bad_words_index:
            if word not in good_and_bad_index:
                if word not in indexes_to_remove:
                    indexes_to_remove.append(word)
    return indexes_to_remove
            
bads = clean_bad_words()
print(bads)



