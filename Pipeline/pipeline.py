from spacy_wrapper import SpaCy
from casen import CasEN
import pandas as pd
from pathlib import Path
import utils
import time
from datetime import datetime

class Pipeline:

    def __init__(self, spaCy:SpaCy=None, casEN:CasEN=None, data:str="", timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.spaCy = spaCy
        self.casEN = casEN

        self.data = data
        self.timer_option = timer_option
        self.log_option = log_option
        self.verbose = verbose
        self.log_location = Path(log_path) if log_path else Path("Logs/log.txt")

    # ---------------------------- TOOLS ----------------------- #
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

    # ---------------------------------- METHODS ---------------------------------- #

    @chrono
    def run(self, spaCy:SpaCy=None, casEN:CasEN=None):

        # ----- INIT ----- #
        data_df = utils.load_data(self.data, self.verbose)

        spaCy = spaCy or self.spaCy
        casEN = casEN or self.casEN
        # ----- SpaCy ----- #
        if spaCy is None:
            spaCy = SpaCy(
                data = data_df,
                timer_option = self.timer_option,
                log_option = self.log_option,
                log_path = self.log_location,
                verbose = self.verbose
            )
        else:
            spaCy.data = data_df

        self.spaCy = spaCy
        # ----- CasEN ----- #
        if casEN is None:
            casEN = CasEN(
                trustable_grf=False,
                verbose=self.verbose
            )
        self.casEN = casEN

        # ------- RUN -------- #
        self.spacy_df = spaCy.run()
        self.casEN_df  = casEN.run()

        # ----- MERGE --- #

        

    