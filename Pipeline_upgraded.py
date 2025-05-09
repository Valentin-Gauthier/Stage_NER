# -------------- IMPORTS ----------------
import time
import pandas as pd
import spacy
import numpy as np
from spacy import displacy
from time import sleep
import bs4
from bs4 import BeautifulSoup
import re
from pathlib import Path
import importlib
from datetime import datetime
import os
# ---------------------------------------
class Pipeline:

    def __init__(self, Excel_raw_data: str, text_column_name: str = "desc",
                 spacy_model: str = "fr_core_news_sm", result_location: str = "Result/xlsx/Pipeline.xlsx",
                 casEN_ipynb_location: str = "../CasEN_fr.2.0/CasEN.ipynb", CasEN_corpus_folder:str="Stage/Result/Corpus", 
                 CasEN_result_folder:str="Stage/Result/CasEN_Result/Res_CasEN_Analyse_synthese_grf",
                 timer: bool = True, timer_log: bool = True, log_location:str="Stage/Result/log",
                 multiple_file_generated: bool = False, auto_correction: bool = True, correction_path:str="Stage/ressources/20231101_Digital3D_Tele-Loisirs _telerama_NER_weekday_evaluation_v3(1).xlsx"):
        
        self.Excel_raw_data = Excel_raw_data
        self.text_column_name = text_column_name
        self.spacy_model = spacy_model
        self.CasEN_result_folder = CasEN_result_folder
        self.CasEN_corpus_folder = CasEN_corpus_folder
        self.NER = None
        self.result_location = result_location
        self.casEN_ipynb_location = casEN_ipynb_location
        self.timer = timer
        self.timer_log = timer_log
        self.log_location = log_location
        self.multiple_file_generated = multiple_file_generated
        self.auto_correction = auto_correction
        self.correction_path = correction_path

        self.df = None
        self.spaCy_model = "fr_core_news_sm"
        
    def chrono(func):
        def wrapper(self, *args, **kwargs):
            if self.timer or self.timer_log:
                start = time.time()
            result = func(self, *args, **kwargs)
            if self.timer or self.timer_log:
                duration = time.time() - start
                if self.timer:
                    print(f"{func.__name__} in : {duration:.2f}s")
                if self.timer_log:
                    self.log_time(func.__name__, duration)
            return result
        return wrapper

    def log_time(self,step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        #check if the path given is a folder
        if os.path.isdir(self.log_location):
            log_file_path = os.path.join(self.log_location, "log.txt")
        else:
            log_file_path = self.log_location


        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - Pipeline [{step}] finish in {duration:.2f} s.\n")

    @chrono
    def load_excel_data(self):
        df = pd.read_excel(self.Excel_raw_data) # open the excel file
        self.df = df.fillna('') # clean dataframe for missing description

    @chrono
    def load_spaCy(self):
        # load the spaCy model
        try:
            importlib.reload(spacy)
            self.NER = spacy.load(self.spacy_model)
        except Exception as e:
            print(f"Error while loading the spaCy '{self.spacy_model}' model. \n{e}")
            raise

    @chrono
    def apply_spacy(self):

        self.ner_spacy = [] # ("text_entite", "label")
        self.ner_spacy_id = [] # ("start", "end")

        # foreach row , use spacy on the description
        for idx, row in self.df.iterrows():
            desc = str(row[str(self.text_column_name)]) # extract the description
            entities = self.NER(desc) # use spacy 
            entities_founds = set() # Manage duplicates automatically
            entities_locations = []

            for word in entities.ents:
                entities_founds.add((word.text, word.label_))
                entities_locations.append((word.start, word.end))

            self.ner_spacy.append(entities_founds)
            self.ner_spacy_id.append(entities_locations)

    @chrono
    def build_spaCy_df(self):

        # Convert the 'titles' and 'desc' columns to lists
        titles_list = self.df["titles"].tolist()
        descriptions_list = self.df[str(self.text_column_name)].tolist()
        
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
            current_NER_doc = self.NER(current_desc)
            
            # Get the precomputed NER entities and their indices for this row
            current_NER = self.ner_spacy[i]
            indices = self.ner_spacy_id[i]
            
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
        
        self.spaCy_df = new_df

    @chrono
    def generate_text_file(self):
        # Remove every file in the folder
        if os.path.exists(self.CasEN_corpus_folder):
            for file in os.listdir(self.CasEN_corpus_folder):
                os.remove(os.path.join(self.CasEN_corpus_folder, file))
                print(f"removed {file} from Corpus")
        else:
            os.makedirs(self.CasEN_corpus_folder)
        # write new file
        if not self.multiple_file_generated:
            with open(f"{self.CasEN_corpus_folder}/corpus.txt", 'w', encoding="utf-8") as f:
                for idx, desc in enumerate(self.df[str(self.text_column_name)]):
                    f.write(f'<doc id="{idx}">')
                    f.write(str(desc))
                    f.write('</doc>\n')

    def remove_casEN_result(self):
        if os.path.exists(self.CasEN_result_folder):
            for file in os.listdir(self.CasEN_result_folder):
                os.remove(os.path.join(self.CasEN_result_folder, file))
            print(f"removed {file} from CasEN result")

    def get_ner_context(self, full_desc: str, ner_text: str, window: int = 10) -> str:
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

    def find_ner_label(self,tag:str) -> str:
        if tag in ["persname", "surname", "forename"]:
            return "PER"
        elif tag in ['placename','geoname', 'adress', 'adrline']:
            return "LOC"
        elif tag in ["orgname", "org"]:
            return "ORG"
        else:
            return "MISC"

    def extract_entities_from_tree(self,soup_doc: BeautifulSoup, doc_id: int):
        """
            Extracts top-level tagged entities (with 'grf' attribute) from an XML/HTML document tree.

            Parameters:
                soup_doc (BeautifulSoup): Parsed document.
                doc_id (int): Identifier of the current document.

            Returns:
                list[dict]: A list of dictionaries representing the extracted entities.
        """
        entities = []
        excluded = {"s", "p", "doc"}

        for element in soup_doc.find_all(lambda tag: tag.name not in excluded and tag.has_attr("grf")):
            if any(ancestor.has_attr("grf") for ancestor in element.parents if ancestor.name not in excluded):
                continue

            main_grf = element["grf"]
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

    @chrono
    def extract_entities_from_file(self, result_path :str):

        with open(result_path, 'r', encoding="utf-8") as f:
            content = f.read()

        content = re.sub(r'</?s\b[^>]*>', '', content) # remove all <s> </s>
        soup = BeautifulSoup(content, 'html.parser')

        all_entities = []

        for doc in soup.find_all('doc'):
            doc_id = int(doc.attrs.get("id"))
            ents = self.extract_entities_from_tree(doc, doc_id)
            all_entities.extend(ents)

        return all_entities

    @chrono
    def build_casEN_df(self):
        list_entities = self.extract_entities_from_file("Result/CasEN_Result/Res_CasEN_Analyse_synthese_grf/corpus.result.txt")
        rows = []
        for entity in list_entities:
            idx = entity["file_id"]
            ner_text = entity["text"]
            context = self.get_ner_context(self.df.loc[idx, "desc"], ner_text, window=7)
            rows.append({
                "titles": self.df.loc[idx, "titles"],
                "NER": ner_text,
                "NER_label": self.find_ner_label(entity["tag"]),
                "desc": context,
                "method": "CasEN",
                "main_graph": entity["grf"],
                "second_graph": entity.get("second_graph", ""),
                "third_graph": entity.get("third_graph", ""),
                "file_id": idx
            })        
        self.casEN_df = pd.DataFrame(rows)

    def determine_method(self,row):
        if row['_merge'] == 'both':
            return "intersection"
        elif row['_merge'] == 'left_only':
            return row['method_spacy']
        elif row['_merge'] == 'right_only':
            return row['method_casEN']
        else:
            return "Spacy"

    @chrono
    def merge_spacy_casEN(self) -> pd.DataFrame:
        """

        """

        if "method" not in self.spaCy_df.columns:
            self.spaCy_df["method"] = "Spacy"

        # Create unique key on ("titles", "NER", "NER_label")
        self.spaCy_df["key"] = self.spaCy_df[["titles", "NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
        self.casEN_df["key"] = self.casEN_df[["titles", "NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)

        # merge dataframs on the key
        merge = pd.merge(self.spaCy_df, self.casEN_df, on='key', how='outer',suffixes=('_spacy', '_casEN'), indicator=True)

        # fix the shared columns
        merge["titles"] = merge["titles_spacy"].combine_first(merge["titles_casEN"])
        merge['method'] = merge.apply(self.determine_method, axis=1)
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

        self.merge = merge[final_columns]

    @chrono
    def correct_excel(self):

        # Créer les clés dans les deux DataFrames
        self.correction_df["key"] = self.correction_df[["titles", "NER", "NER_label", "method", "hash"]].apply(tuple, axis=1)
        self.merge["key"] = self.merge[["titles", "NER", "NER_label", "method", "file_id"]].apply(tuple, axis=1)

        # Colonnes à transférer
        cols_to_copy = ["manual cat", "correct", "extent", "category"]

        # Créer un dictionnaire : clé -> valeurs des colonnes à copier
        correction_dict = self.correction_df.set_index("key")[cols_to_copy].to_dict(orient="index")

        # Pour chaque ligne de self.merge, copier si la clé existe
        for col in cols_to_copy:
            self.merge[col] = self.merge["key"].map(lambda k: correction_dict.get(k, {}).get(col, ""))

        # Supprimer la clé temporaire
        self.merge.drop(columns=["key"], inplace=True)

        return self.merge

    @chrono
    def run(self):
        #load the data
        self.load_excel_data()

        # ---------- spaCy -----------
        self.load_spaCy()
        self.apply_spacy()
        self.build_spaCy_df()
        # ---------- CasEN ----------
        self.generate_text_file()
        # remove every file in the casEN result folder
        self.remove_casEN_result()
        get_ipython().run_line_magic('run', str(self.casEN_ipynb_location))
        self.build_casEN_df()

        # ----------------- Merge Result ------------------------------
        self.merge_spacy_casEN()
        
       
        if self.auto_correction: # apply correction
            self.correction_df = pd.read_excel(self.correction_path)
            self.correct_excel()

        self.merge.to_excel(self.result_location, index=False) # Converte into Excel File
 
        
if __name__ == "__main__":
    data = "Stage/ressources/20231101_raw.xlsx"
    pipeline = Pipeline(Excel_raw_data=data)
    pipeline.run()

