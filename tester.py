import pandas as pd
import re

def replace(old, new, full_text):
    return re.sub(re.escape(old), new, full_text, flags=re.IGNORECASE)

bad_words = [" vada", "vadný", "vadny", "vadné", "vadne", "vadná", " rozbit", " havarovan", " poškozen", " poskozen", "špatn", "nepojízd", "nepojizdn", 
" bourané", " bourane", " bouraný", " bourany", "koroze", "kosmetick", "dodělaní", "na náhradní díly", "na nahradni dily", "porucha", " porouchan", " KO!",
"drobné závady", "zavady", "závad", "oděrky", "zreziv", "rezav", "přetržený", "pretrzeny", "praskl", "nenastartuje", "nenaskočí", "problém s", "netopi", "netopí", "nejede",
"zreziv", " vada", "po výměmě motoru", "odřeniny", "promacknut", "promáčknut", "neřadí"]
good_words = ["bez poškození", "žádné poškození", "nemá poškození", "není poškozen", "bez koroze", 
"žádné závady", "bez závad"]

bad_without_good_words = []
for word in bad_words:
    if word not in good_words:
        bad_without_good_words.append(word)

data = [["price info clean", "additional info wrong vadný", "detail info good bez závad"],
        ["rezavý", "bez koroze", "detail info clean"],
        ["bez koroze", "bez závad", "čistý"],
        ["detail poškozen", "additional info clean", "detail info clean"],
        ["detail clean 1", "additional info clean 2", "detail info clean 3"]]

raw_data_updated = pd.DataFrame(data, columns = ["price_more_info", "additional_info", "detail"])

print(raw_data_updated.head())

def clean_bad_words():
    indexes_to_remove = []
    for cat in ["price_more_info", "additional_info", "detail"]:
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
print(f"\nindex to drop: {index_to_drop}\n")
raw_data_updated = raw_data_updated.drop(index=index_to_drop, axis=0)

pd.set_option('display.max_colwidth', None)

print(raw_data_updated.head())
