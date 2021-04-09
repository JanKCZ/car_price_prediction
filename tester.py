import pandas as pd
import numpy as np
from timeit import timeit



raw_data_updated = pd.read_csv('/Users/jankolnik/Downloads/car_list_all_v1_updated_sauto.csv', dtype=str)

#remove adds, which include words about damaged or non-functional cars
bad_words = [" vadný", " vadny", " vadné", " vadne", " rozbit", " havarovan", " poškozen", " poskozen", "špatn", "nepojízd", "nepojizdn", 
" bourané", " bourane", " bouraný", " bourany", "koroze", "kosmetick", "dodělaní", "na náhradní díly", "na nahradni dily", "porucha", " porouchan", " KO!",
"drobné závady", "zavady", "závad", "oděrky", "zreziv", "rezav", "přetržený", "pretrzeny", "praskl", "nenastartuje", "nenaskočí", "problém s", "netopi", "netopí", "nejede",
"zreziv", " vada"]
good_words = ["bez poškození", "žádné poškození", "nemá poškození", "není poškozen", "bez koroze", 
"žádné závady", "bez závad"]
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


remove_bad_words()
remove_bad_words_comprehention()
            
# not_nan = raw_data_updated[raw_data_updated.additional_info.notnull()]
# for word in bad_words:
# bad_words_index = not_nan[not_nan.additional_info.str.contains(word, case = False)].index
# for good in good_words:
#     if good not in bad_words_index:
#     for index in bad_words_index:
#         if index not in bad_index:
#         bad_index.append(index)

# not_nan = raw_data_updated[raw_data_updated.detail.notnull()]
# for word in bad_words:
# bad_words_index = not_nan[not_nan.detail.str.contains(word, case = False)].index
# for good in good_words:
#     if good not in bad_words_index:
#     for index in bad_words_index:
#         if index not in bad_index:
#         bad_index.append(index)
        
# raw_data_updated = raw_data_updated.drop(bad_index)