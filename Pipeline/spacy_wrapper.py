import pandas as pd
import spacy
from pathlib import Path
import utils
import time
from datetime import datetime

class SpaCy():

    def __init__(self, data:str=None, spaCy_model:str="fr_core_news_sm", make_excel:str=None, timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.data = data
        self.spaCy_model = spaCy_model
        self.timer_option = timer_option
        self.log_option = log_option
        self.verbose = verbose
        self.log_path = log_path
        self.make_excel = make_excel

        # -------------- Init -------------- #
        self.nlp = spacy.load(spaCy_model) # Load the spaCy model
        self.log_location = Path(log_path) if log_path else Path("Logs/log.txt")

        # if the file is given then load it
        if isinstance(self.data, str):
            self.data = utils.load_data(data=self.data, verbose=self.verbose)

    # ---------------------------- TOOLS ----------------------- #
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_location.is_dir(): 
            log_file_path = self.log_location / "log.txt" # if the path is a folder -> add a filename
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

    # ---------------------------------- METHODS ---------------------------------- #
    @chrono
    def run(self, verbose:bool=None) -> pd.DataFrame:

        if verbose is None:
            verbose = self.verbose

        if verbose:
            print(f"[spaCy] spaCy version: {spacy.__version__}")
            print(f"[spaCy] spaCy model: {self.nlp.meta.get('name', 'unknown')}")

        rows = []
        for idx, df_row in self.data.iterrows():
            doc = self.nlp(df_row["desc"])
            for ent in doc.ents:
                start = max(0, ent.start_char - 40)
                end = ent.end_char + 40
                rows.append({
                    "titles" : self.data.loc[idx, "titles"],
                    "NER" : ent.text,
                    "NER_label" : ent.label_,
                    "desc" : df_row["desc"][start:end],
                    "method": "spaCy",
                    "file_id" : idx
                })

        self.df = pd.DataFrame(rows)

        if self.make_excel is not None:
            self.df.to_excel(f"{self.make_excel}.xlsx")

        if verbose:
            print("SpaCy run is finish")

        return self.df