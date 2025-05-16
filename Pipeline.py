import pandas as pd
import spacy
import time
from datetime import datetime
from pathlib import Path
import shutil
import re
from bs4 import BeautifulSoup
import json

class Pipeline:

    def __init__(self, 
                 Excel_files:str, 
                 timer_option:bool=True, 
                 log_option:bool=True,
                 log_destination:str="Result/log",
                 spaCy_model:str="fr_core_news_sm",
                 include_casEN_MISC : bool = True,
                 remove_duplicate_rows : bool = False,
                 casEN_ipynb_location:str="../CasEN_fr.2.0/CasEN.ipynb",
                 casEN_corpus_folder:str="Result/Corpus",
                 casEN_corpus_unique:bool=True,
                 casEN_result_folder:str="Result/CasEN_Result/Res_CasEN_Analyse_synthese_grf",
                 Excel_result_path : str = "Result/xlsx/",
                 correction_path : str = None,
                 allowed_grf:str=None
                 ):
       
        self.timer_option = timer_option  # display the time
        self.log_option = log_option  # write every return in log file
        self.log_location = Path(log_destination) if log_destination else Path("logs/log.txt")
        self.nlp = spacy.load(spaCy_model)  # natural language processing
        self.include_casEN_MISC = include_casEN_MISC
        self.remove_duplicate_rows = remove_duplicate_rows
        self.casEN_ipynb_location = casEN_ipynb_location # the code for running casEN
        self.casEN_corpus_folder = casEN_corpus_folder # the path for the corpus folder
        self.casEN_corpus_unique = casEN_corpus_unique # choose if we want generate unique or multiple files before casEN analyse
        self.casEN_result_folder = casEN_result_folder # the path of the result files by casEN
        self.Excel_result_path = Excel_result_path # the location of the Excel file product by the Pipeline
        self.correction_path = correction_path # THe path for the correction(if None, no correction)
        self.allowed_grf = allowed_grf # the path for the JSON file wich contains graphs we want to keep in only casEN entities

        self.load_excel(Excel_files) # Load the self.df


    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_location.is_dir(): 
            log_file_path = self.log_location / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_location


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - Pipeline [{step}] finish in {duration:.2f} s.\n")

    def chrono(func):
        def wrapper(self, *args, **kwargs):
            if self.timer_option or self.log_option:
                start = time.time()
            result = func(self, *args, **kwargs)
            if self.timer_option or self.log_option:
                duration = time.time() - start
                if self.timer_option:
                    print(f"{func.__name__} in : {duration:.2f}s")
                if self.log_option:
                    self.log(func.__name__, duration)
            return result
        return wrapper

    @chrono
    def load_excel(self, Excel_files:str) -> pd.DataFrame:
        excel_path = Path(Excel_files)
        if not excel_path.is_file():
            raise FileNotFoundError(f"The Excel file doesn't exist : {excel_path}")
        self.df = pd.read_excel(excel_path)
        self.df["desc"] = self.df["desc"].fillna("") # clean the empty description
        return self.df

    @chrono
    def use_spaCy(self, verbose:bool=False) -> pd.DataFrame:
        """
            Applies spaCy's Named Entity Recognition (NER) model to the 'desc' column of the DataFrame.
            spacy documentation : https://spacy.io/

            Returns:
                pd.DataFrame: new DataFrame containing one row per detected entity with columns :
                    - 'titles': The original title associated with the row.
                    - 'NER': The extracted named entity.
                    - 'NER_label': The entity type label from spaCy (e.g., 'PER', 'LOC').
                    - 'desc': A text excerpt surrounding the entity for context.
                    - 'method': The string 'spaCy', indicating the method used.
                    - 'file_id': The index of the original row in the source DataFrame.
        """
        if verbose:
            print(f"[spaCy] spaCy version: {spacy.__version__}")
            print(f"[spaCy] spaCy model: {self.nlp.meta.get('name', 'unknown')}")

        rows = []
        for idx, df_row in self.df.iterrows():
            doc = self.nlp(df_row["desc"])
            for ent in doc.ents:
                start = max(0, ent.start_char - 40)
                end = ent.end_char + 40
                rows.append({
                    "titles" : self.df.loc[idx, "titles"],
                    "NER" : ent.text,
                    "NER_label" : ent.label_,
                    "desc" : df_row["desc"][start:end],
                    "method": "spaCy",
                    "file_id" : idx
                })

        self.spaCy_df = pd.DataFrame(rows)
        return self.spaCy_df

    @chrono
    def execute_casEN(self, verbose:bool=False):
        self.prepare_folder(self.casEN_result_folder, verbose)
        get_ipython().run_line_magic('run', str(self.casEN_ipynb_location))
            
    def prepare_folder(self, folder_path:str, verbose:bool=False):
        folder = Path(folder_path)
        # check if the path is good
        if not  folder.exists():
            raise FileNotFoundError(f"casEN corpus folder does not exist : {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Provided casEN path is not a folder : {folder}")
        if verbose:
            print(f"[casEN] folder : {folder}")

        # removed everything in the folder
        files = list(folder.iterdir())
        if not files:
            if verbose:
                print(f"[casEN] Folder is empty")
        else:
            for file in files:
                try:
                    if file.is_file() or file.is_symlink():
                        file.unlink()
                        if verbose:
                            print(f"[casEN] Deleted file : {file.name}")
                    elif file.is_dir():
                        shutil.rmtree(file)
                        if verbose:
                            print(f"[casEN] Deleted folder : {file.name}")
                except Exception as e:
                    print(f"[casEN] Failed to delete {file} : {e}")

    @chrono
    def generate_casEN_file(self, unique_file: bool = None, verbose: bool = False):
        # Prepare the folder
        self.prepare_folder(self.casEN_corpus_folder, verbose)

        if unique_file is None:
            unique_file = self.casEN_corpus_unique

        if unique_file:
            # Case 1: Generate a single corpus.txt file
            missing_desc = (self.df["desc"] == "").sum()
            corpus_path = Path(self.casEN_corpus_folder) / "corpus.txt"
            with open(corpus_path, 'w', encoding="utf-8") as f:
                for idx, row in self.df.iterrows():
                    f.write(f'<doc id="{idx}">')
                    f.write(str(row["desc"]))
                    f.write('</doc>\n')
            if verbose:
                print(f"[prepare folder] Single corpus file generated: {corpus_path}")
                print(f"[prepare folder] Missing description : {missing_desc}")
        else:
            # Case 2: Generate one file per row
            for idx, row in self.df.iterrows():
                doc_path = Path(self.casEN_corpus_folder) / f"doc_{idx}.txt"
                with open(doc_path, 'w', encoding="utf-8") as f:
                    f.write(f'<doc id="{idx}">')
                    f.write(str(row["desc"]))
                    f.write('</doc>\n')
            if verbose:
                print(f"[prepare folder] {len(self.df)} individual corpus files generated in {self.casEN_corpus_folder}")
                print(f"[prepare folder] Missing description : {missing_desc}")

    def extract_entities_from_desc(self, soup_doc : BeautifulSoup, doc_id:int) -> list[dict]:
        entities = []
        excluded = ["s", "p", "doc"]
        for element in soup_doc.find_all(lambda tag: tag not in excluded and tag.has_attr("grf")):
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
    def extract_casEN_entities(self, casEN_result_folder:str=None) -> list[dict]:
        if casEN_result_folder is None:
            casEN_result_folder = self.casEN_result_folder

        all_entities = []

        if self.casEN_corpus_unique:
            # One file
            file_path = Path(casEN_result_folder) / "corpus.result.txt"
            with open(file_path, 'r', encoding="utf-8") as f:
                content = f.read()

            content = re.sub(r'</?s\b[^>]*>', '', content)
            soup = BeautifulSoup(content, 'html.parser')
            for doc in soup.find_all("doc"):
                doc_id = int(doc.attrs.get("id"))
                ents = self.extract_entities_from_desc(doc, doc_id)
                all_entities.extend(ents)
        else:
            # Multiple file
            result_folder = Path(casEN_result_folder)
            for file in result_folder.glob("*.txt"):
               
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()

                content = re.sub(r'</?s\b[^>]*>', '', content)
                soup = BeautifulSoup(content, 'html.parser')
                for doc in soup.find_all("doc"):
                    doc_id = int(doc.attrs.get("id"))
                    ents = self.extract_entities_from_desc(doc, doc_id)
                    all_entities.extend(ents)

        return all_entities

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

    def find_ner_label(self, tag:str) -> str:
        if tag in ["persname", "surname", "forename"]:
            return "PER"
        elif tag in ['placename','geoname', 'adress', 'adrline']:
            return "LOC"
        elif tag in ["orgname", "org"]:
            return "ORG"
        else:
            return "MISC"

    @chrono
    def use_casEN(self, verbose:bool=False) -> pd.DataFrame:
        self.generate_casEN_file(self.casEN_corpus_unique, verbose)
        self.execute_casEN(verbose)
        
        entities_list = self.extract_casEN_entities()
        rows = []
        for entity in entities_list:
            ner_label =  self.find_ner_label(entity["tag"])
            if  not self.include_casEN_MISC and ner_label == "MISC":
                continue
            idx = entity["file_id"]
            ner_text = entity["text"]
            context = self.get_ner_context(self.df.loc[idx, "desc"], ner_text, window=7)
            rows.append({
                "titles": self.df.loc[idx, "titles"],
                "NER" : ner_text,
                "NER_label":ner_label,
                "desc" : context,
                "method" : "casEN",
                "main_graph": entity["grf"],
                "second_graph" : entity.get("second_graph", ""),
                "third_graph" : entity.get("third_graph", ""),
                "file_id" : idx
            })
            self.casEN_df = pd.DataFrame(rows)
        return self.casEN_df

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
    def merge_spacy_casEN(self, spaCy_df:pd.DataFrame=None, casEN_df:pd.DataFrame=None, verbose:bool=False) -> pd.DataFrame:
        """

        """
        if spaCy_df is None:
            spaCy_df = self.spaCy_df
        if casEN_df is None:
            casEN_df = self.casEN_df

        if verbose == True:
            df_non_vides = self.df[self.df["desc"] != ""]

        # Create unique key on ("titles", "NER", "NER_label")
        spaCy_df["key"] = spaCy_df[["titles", "NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)
        casEN_df["key"] = casEN_df[["titles", "NER", "NER_label", "file_id"]].apply(lambda x: tuple(x), axis=1)

        # merge dataframs on the key
        merge = pd.merge(spaCy_df, casEN_df, on='key', how='outer',suffixes=('_spacy', '_casEN'), indicator=True)

        # fix the shared columns
        merge["titles"] = merge["titles_spacy"].combine_first(merge["titles_casEN"])
        merge['method'] = merge.apply(self.determine_method, axis=1)
        merge["NER"] = merge["NER_spacy"].combine_first(merge["NER_casEN"])
        merge["NER_label"] = merge["NER_label_spacy"].combine_first(merge["NER_label_casEN"])
        merge["desc"] = merge["desc_spacy"].combine_first(merge["desc_casEN"])
        merge["file_id"] = merge["file_id_spacy"].combine_first(merge["file_id_casEN"])
        
        merge = merge.sort_values(
            by=["file_id"], 
            ascending=[True]
        ).reset_index(drop=True)

        merge["manual cat"] = ""
        merge["extent"] = ""
        merge["correct"] = ""
        merge["category"] = ""

        final_columns = ['manual cat', 'correct', 'extent', 'category','titles', 'NER', 'NER_label', 'desc', 'method',
                        'main_graph', 'second_graph', 'third_graph', "file_id"]

        if self.remove_duplicate_rows:
            merge = merge.drop_duplicates(subset=["titles", "NER", "NER_label", "method", "main_graph", "second_graph", "third_graph"])

        self.merge = merge[final_columns]

        if verbose == True:
            files_with_entities = set(merge["file_id"])
            total_with_desc = len(df_non_vides)
            desc_without_entity = total_with_desc - len(files_with_entities)
            print(f"[merge] description without entities : {desc_without_entity}")


        return self.merge

    @chrono
    def optimisation(self, merge:pd.DataFrame=None,allowed_grf:list=None, verbose:bool=False) -> pd.DataFrame:
            """
            
                Preserves the maximum amount of data while maintaining the best possible accuracy

                Parameters:
                    merge (pd.DataFrame): result of the merger between spaCy and casEN 

                    allowed_grf (str): the path for the  JSON file contains graphs

                Returns:
                    pd.DataFrame 
            """
            
            if merge is None:
                merge = self.merge
            if allowed_grf is None:
                allowed_grf = self.allowed_grf

            with open(allowed_grf, 'r', encoding="utf-8") as f:
                allowed = json.load(f)

            def is_allowed(row):
                for combo in allowed:
                    if all(row.get(col) == val for col, val in combo.items()):
                        return True
                return False

            if verbose:
                print("[opt] count par méthode avant :", merge["method"].value_counts(dropna=False))

            mask = merge.apply(
                lambda row: (
                    row["method"] == "intersection"
                    or (row["method"] == "casEN" and is_allowed(row))
                ),
                axis=1
            )
            if verbose:
                print("[opt] count par méthode après :", merge.loc[mask, "method"].value_counts(dropna=False))

            self.merge = merge.loc[mask].reset_index(drop=True)

            return self.merge

    @chrono
    def correct_excel(self, verbose:bool=False):

        correction_df = pd.read_excel(self.correction_path)

        correction_df["key"] = correction_df[["NER", "NER_label", "hash"]].apply(tuple, axis=1)
        self.merge["key"] = self.merge[["NER", "NER_label", "file_id"]].apply(tuple, axis=1)

        cols_to_copy = ["manual cat", "correct", "extent", "category"]

        correction_dict = correction_df.set_index("key")[cols_to_copy].to_dict(orient="index")

        for col in cols_to_copy:
            self.merge[col] = self.merge["key"].map(lambda k: correction_dict.get(k, {}).get(col, ""))

        self.merge.drop(columns=["key"], inplace=True)

        return self.merge
    
    @chrono
    def run(self, verbose: bool = False) -> str:
        self.use_spaCy(verbose=verbose)
        self.use_casEN(verbose=verbose)
        self.merge_spacy_casEN(verbose=verbose)

        if self.allowed_grf != None:
            self.optimisation(verbose=verbose)

        if self.correction_path != None:
            self.correct_excel(verbose=verbose)

        base_filename = Path(self.Excel_result_path) / "Pipeline.xlsx"
        filename = base_filename
        counter = 1

        # Check if file already exists
        while filename.exists():
            filename = base_filename.with_stem(f"{base_filename.stem}({counter})")
            counter += 1

        self.merge.to_excel(filename, index=False)
        return f"Excel file saved as {filename}."







