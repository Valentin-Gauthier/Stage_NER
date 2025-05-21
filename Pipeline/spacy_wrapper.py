import time
from datetime import datetime
import pandas as pd
import spacy
from pathlib import Path

class SpaCy():

    def __init__(self, data:str=None, spaCy_model:str="fr_core_news_sm", timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.data = data
        self.spaCy_model = spaCy_model
        self.timer_option = timer_option
        self.log_option = log_option
        self.verbose = verbose
        self.log_path = log_path

        # -------------- Init -------------- #
        self.nlp = spacy.load(spaCy_model) # Load the spaCy model
        self.log_location = Path(log_path) if log_path else Path("Logs/log.txt")

        if self.data != None:
            self.load_data(self.verbose)
        


    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_location.is_dir(): 
            log_file_path = self.log_location / "log_spacy.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_location


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - SpaCy [{step}] finish in {duration:.2f} s.\n")

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
    def load_data(self, verbose:bool=None) -> pd.DataFrame:
        if verbose is None:
            verbose = self.verbose

        excel_path = Path(self.data)
        if not excel_path.is_file():
            raise FileNotFoundError(f"The Excel file doesn't exist : {excel_path}")
        self.df = pd.read_excel(excel_path)
        self.df["desc"] = self.df["desc"].fillna("") # clean the empty description
        return self.df

    @chrono
    def run(self, verbose:bool=None) -> pd.DataFrame:
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
        if verbose is None:
            verbose = self.verbose

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
