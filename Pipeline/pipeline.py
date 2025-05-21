from spacy_wrapper import SpaCy
from casen import CasEN
import time
from datetime import datetime
import pandas as pd

class Pipeline:

    def __init__(self, spaCy:SpaCy=None, casEN:CasEN=None):
        self.spaCy = spaCy
        self.casEN = casEN


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



    def run(self, spaCy:SpaCy=None, casEN:CasEN=None):

        if spaCy is None:
            spaCy = self.spaCy
        if casEN is None:
            casEN = self.casEN

        spaCy.run()
        casEN.run()
        

    