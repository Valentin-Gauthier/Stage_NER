# -------------- IMPORTS ----------------
import time
import pandas as pd
import spacy
import numpy as np
from spacy import displacy
from time import sleep
import bs4
from bs4 import BeautifulSoup
import os
import requests
import re
#import lxml
import json
from pathlib import Path
# ---------------------------------------


# ---------------------- LOAD DATAS -------------------
def load_excel_file(file_path:str):


    """
        Load a Excel file into a DataFrame

        Parameters:
            file_path (str): The file path

        Returns:
            df (pd.Dataframe) : the result of Excel file
    """

    # Start Chrono
    start_chrono = time.time()

    # Open the Excel file
    df = pd.read_excel(file_path)

    # Clean
    df.drop(df.columns[0], axis=1, inplace=True) # Drop the Column
    #print(df.isnull().sum())
    df = df.fillna('') # clean the missing description
    
    # End Chrono
    end_chrono = time.time()
    chrono = end_chrono - start_chrono
    print(f"Load data in : {chrono:.2f}s")

    return df

# ------------------------ SPACY -----------------------

def apply_spacy(df, NER):
    """
        Apply Spacy

        Parameters:
            df (pd.Dataframe): the dataframe 
    """

    ner_spacy = [] # ("text_entite", "label")
    ner_spacy_id = [] # ("start", "end")

    # Start Chrono
    start_chrono = time.time()
    # foreach row , use spacy on the description
    for idx, row in df.iterrows():
        desc = str(row["desc"]) # extract the description
        entities = NER(desc) # use spacy 
        entities_founds = set() # Manage duplicates automatically
        entities_locations = []

        for word in entities.ents:
            entities_founds.add((word.text, word.label_))
            entities_locations.append((word.start, word.end))

        ner_spacy.append(entities_founds)
        ner_spacy_id.append(entities_locations)

    # End Chrono
    end_chrono = time.time()
    chrono = end_chrono - start_chrono
    print(f"Apply Spacy in : {chrono:.2f}s")

    return (ner_spacy, ner_spacy_id)

def build_ner_dataframe(df, ner_spacy, ner_spacy_id, NER):
    """
    Build a new DataFrame from the original DataFrame and precomputed NER outputs.
    
    The original DataFrame must contain at least two columns: 'titles' and 'desc'.
    For each row, the function uses the provided NER results (entity text/label and their corresponding indices)
    to extract a portion of the description text with some context around each entity.
    
    Parameters:
        df (pd.DataFrame): The original DataFrame containing a 'titles' column and a 'desc' (description) column.
        ner_spacy (list): A list (one per row) of lists of tuples, each tuple containing the entity text and its label.
                          For example: [[("Barack Obama", "PERSON"), ("Washington", "GPE")], ...]
        ner_spacy_id (list): A list (one per row) of lists of tuples, each tuple containing the start and end indices of the entity.
                             For example: [[(0, 12), (20, 30)], ...]
                             
    Returns:
        pd.DataFrame: A new DataFrame with the following columns:
                      - "titles": the original title repeated for each extracted entity.
                      - "NER": the extracted entity text.
                      - "NER_label": the label/type of the entity.
                      - "desc": a snippet of the original description around the entity.
    """

    start_chrono = time.time()

    # Convert the 'titles' and 'desc' columns to lists
    titles_list = df["titles"].tolist()
    descriptions_list = df["desc"].tolist()
    
    # Initialize lists to store the new data
    title_list = []
    text_list = []
    label_list = []
    location_list = []
    file_id_list = []
    
    # Loop through each entry in the original DataFrame (by index)
    for i in range(len(titles_list)):
        current_desc = descriptions_list[i]
        # Process the description using the NER function to obtain a full processed document.
        # Assume that NER() returns an object where converting it to a string gives its full text.
        current_NER_doc = NER(current_desc)
        
        # Get the precomputed NER entities and their indices for this row
        current_NER = ner_spacy[i]
        indices = ner_spacy_id[i]
        
        # Get the title associated with the current row
        current_title = titles_list[i]
        
        # Loop over each entity and its corresponding indices
        for entity, idx in zip(current_NER, indices):
            entity_text, entity_label = entity
            
            # Append data to the corresponding list
            title_list.append(current_title)
            text_list.append(entity_text)
            label_list.append(entity_label)
            file_id_list.append(i)
            
            # Extract a snippet from the full document text around the entity for context.
            # If the entity starts near the beginning, we extract from the start.
            if idx[0] < 10:
                # If entity is close to the end of the document, include all text until the end.
                if idx[1] + 8 > len(current_NER_doc):
                    location_list.append(current_NER_doc)
                else:
                    location_list.append(current_NER_doc[:idx[1] + 8])
            else:
                # Else extract 10 characters before the entity start.
                # If the entity is near the end, extract until the end.
                if idx[1] + 8 > len(current_NER_doc):
                    location_list.append(current_NER_doc[idx[0] - 10:])
                else:
                    location_list.append(current_NER_doc[idx[0] - 10: idx[1] + 8])
    
    # Build a new DataFrame using the collected lists.
    new_df = pd.DataFrame(zip(title_list, text_list, label_list, location_list, file_id_list))
    
    # Rename the columns to be more informative.
    new_df.rename(columns={0: "titles", 1: "NER", 2: "NER_label", 3: "desc", 4:"file_id"}, inplace=True)

    end_chrono = time.time()
    chrono = end_chrono - start_chrono
    print(f"Generate Spacy DataFrame in : {chrono:.2f}s")
    
    return new_df

# ------------------------ CASEN -----------------------

def generate_text_files(df):
    start_chrono = time.time()

    # Generate text file 
    text_location = "Result/Corpus/All.txt"
    
    with open(text_location, 'w', encoding="utf-8") as f:
        for idx, desc in enumerate(df["desc"]):
            f.write(f'<doc id="{idx}">')
            f.write(str(desc))
            f.write('</doc>\n')

    end_chrono = time.time()
    chrono = end_chrono - start_chrono
    print(f"Generate text files in : {chrono:.2f}s")

# Extract data From CasEN file

def extract_entities_from_tree(soup_doc: BeautifulSoup, doc_id: int):
    """
    Ne remonte que les entités ayant un grf et sans ancêtre grf,
    et stocke pour chacune :
      - file_id, tag, text, grf (main)
      - second_graph  = grf du 1er enfant direct
      - third_graph   = grf du 2ème enfant direct
    """
    entities = []
    excluded = {"s", "p", "doc"}

    # On parcourt tous les éléments taggés avec grf
    for element in soup_doc.find_all(lambda tag: tag.name not in excluded and tag.has_attr("grf")):
        # Si un des ancêtres a aussi un grf, on l'ignore (c'est un sous-élément)
        if any(ancestor.has_attr("grf") for ancestor in element.parents if ancestor.name not in excluded):
            continue

        main_grf = element["grf"]
        # on collecte les grf des enfants directs pour second et third
        children = [child for child in element.find_all(recursive=False) if child.has_attr("grf")]
        second = children[0]["grf"] if len(children) >= 1 else ""
        third  = children[1]["grf"] if len(children) >= 2 else ""

        entities.append({
            "file_id":       doc_id,
            "tag":           element.name,
            "text":          element.get_text(),
            "grf":           main_grf,
            "second_graph":  second,
            "third_graph":   third
        })

    return entities

def extract_entities_from_file(result_path :str):

    with open(result_path, 'r', encoding="utf-8") as f:
        content = f.read()

    content = re.sub(r'</?s\b[^>]*>', '', content) # remove all <s> </s>
    soup = BeautifulSoup(content, 'html.parser')

    all_entities = []

    for doc in soup.find_all('doc'):
        doc_id = int(doc.attrs.get("id"))
        ents = extract_entities_from_tree(doc, doc_id)
        all_entities.extend(ents)

    return all_entities

def casEN_df_single(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a DataFrame of NER results from a single CasEN output file.
    """
    start = time.time()

    list_entities = extract_entities_from_file("Result/CasEN_Result/Res_CasEN_Analyse_synthese_grf/All.result.txt")
    rows = []
    for entity in list_entities:
        idx = entity["file_id"]
        ner_text = entity["text"]
        context = get_ner_context(df.loc[idx, "desc"], ner_text, window=7)
        rows.append({
            "titles": df.loc[idx, "titles"],
            "NER": ner_text,
            "NER_label": find_ner_label(entity["tag"]),
            "desc": context,
            "method": "CasEN",
            "main_graph": entity["grf"],
            "second_graph": entity.get("second_graph", ""),
            "third_graph": entity.get("third_graph", ""),
            "file_id": idx
        })

    df_casen = pd.DataFrame(rows)
    print(f"Built CasEN DataFrame in {time.time() - start:.2f}s")
    return df_casen

def get_ner_context(full_desc: str, ner_text: str, window: int = 10) -> str:
    """
    Extracts a context window around the first match of the NER text in the full descriptionE
    
    Parameters:
        full_desc (str): The full text description.
        ner_text (str): The NER text to find in the description.
        window (int): Number of words to extract before and after the match

    Returns:
        str: A snippet of the description centered around the NER match
    """
    # Normalize input for search
    full_desc_clean = re.sub(r'[^\w\s]', '', full_desc.lower())
    ner_clean = re.sub(r'[^\w\s]', '', ner_text.lower())

    # Tokenize words
    words = full_desc.split()
    clean_words = full_desc_clean.split()
    ner_words = ner_clean.split()

    # Try to find full match of all NER words
    for i in range(len(clean_words) - len(ner_words) + 1):
        if clean_words[i:i+len(ner_words)] == ner_words:
            start = max(0, i - window)
            end = min(len(words), i + len(ner_words) + window)
            return ' '.join(words[start:end])

    return full_desc

def find_ner_label(tag:str) -> str:
    if tag in ["persname", "surname", "forename"]:
        return "PER"
    elif tag in ['placename','geoname', 'adress', 'adrline']:
        return "LOC"
    elif tag in ["orgname", "org"]:
        return "ORG"
    else:
        return "MISC"

# ---------------------- MERGE ------------------------
def determine_method(row):
    if row['_merge'] == 'both':
        return "intersection"
    elif row['_merge'] == 'left_only':
        return row['method_spacy']
    elif row['_merge'] == 'right_only':
        return row['method_casEN']
    else:
        return "Spacy"

def merge_spacy_casEN(df_spacy:pd.DataFrame,df_casEN:pd.DataFrame) -> pd.DataFrame:
    """

    
    """
    start_chrono = time.time()

    if "method" not in df_spacy.columns:
        df_spacy["method"] = "Spacy"

    # Create unique key on ("titles", "NER", "NER_label")
    df_spacy["key"] = df_spacy[["titles", "NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
    df_casEN["key"] = df_casEN[["titles", "NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)

    # merge dataframs on the key
    merge = pd.merge(df_spacy, df_casEN, on='key', how='outer',suffixes=('_spacy', '_casEN'), indicator=True)

    # fix the shared columns
    merge["titles"] = merge["titles_spacy"].combine_first(merge["titles_casEN"])
    merge['method'] = merge.apply(determine_method, axis=1)
    merge["NER"] = merge["NER_spacy"].combine_first(merge["NER_casEN"])
    merge["NER_label"] = merge["NER_label_spacy"].combine_first(merge["NER_label_casEN"])
    merge["desc"] = merge["desc_casEN"].combine_first(merge["desc_spacy"])
    merge["file_id"] = merge["file_id_casEN"].combine_first(merge["file_id_spacy"])
    
    merge = merge.sort_values(
        by=["titles"], 
        ascending=[False]
    ).reset_index(drop=True)

    merge["manual cat"] = ""
    merge["extent"] = ""
    merge["correct"] = ""
    merge["category"] = ""

    final_columns = ['manual cat', 'correct', 'extent', 'category','titles', 'NER', 'NER_label', 'desc', 'method',
                     'main_graph', 'second_graph', 'third_graph', "file_id"]


    merge = merge.drop_duplicates(subset=["titles", "NER", "NER_label", "method", "main_graph", "second_graph", "third_graph"])

    end_chrono = time.time()
    chrono = end_chrono - start_chrono
    print(f"Merge CasEN & Spacy in : {chrono:.2f}s")

    return merge[final_columns]

def Pipeline(data_path:str, excel_result:str, spacy_model:str="fr_core_news_sm"):
    """
        Run everything in one function

    """
    print(f"spacy model :{spacy_model}")
    
    # Start Chrono
    start_chrono = time.time()
    # Data to load
    df = load_excel_file(data_path)

    # ----------------------- Use Spacy ----------------------------
    NER = spacy.load(spacy_model)
    ner_spacy, ner_spacy_id = apply_spacy(df, NER)
    df_spacy = build_ner_dataframe(df, ner_spacy, ner_spacy_id, NER)

    # ----------------------- Use CasEN ----------------------------
    generate_text_files(df)
    # Run CasEN
    casen_ipynb_path = "../CasEN_fr.2.0/CasEN.ipynb"
    get_ipython().run_line_magic('run', str(casen_ipynb_path))
    df_casEN = casEN_df_single(df)

    # ----------------- Merge Result ------------------------------
    intersection = merge_spacy_casEN(df_spacy, df_casEN)
    
    # Converte into Excel File
    intersection.to_excel(excel_result, index=False)
    

    # End Chrono
    end_chrono = time.time()
    chrono = end_chrono - start_chrono
    print(f"Total Execute in : {chrono:.2f}s")

