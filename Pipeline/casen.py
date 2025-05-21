import utils
import time
from datetime import datetime

class CasEN():

    def __init__(self, path:str="", trustable_grf:bool=False, archiving:bool=False, make_excel:bool=False, timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.path = path
        self.trustable_grf = trustable_grf
        self.timer_option = timer_option
        self.log_option = log_option
        self.verbose = verbose
        self.log_path = log_path
        self.make_excel = make_excel


    # ---------------------------- TOOLS ----------------------- #
    def log(self, step:str, duration:float):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

        if self.log_location.is_dir(): 
            log_file_path = self.log_location / "log.txt" # if the path is a folder -> add a filename
        else:
            log_file_path = self.log_location


        log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        with open(log_file_path, 'a', encoding="utf-8") as f:
            f.write(f"{timestamp} - CasEN [{step}] finish in {duration:.2f} s.\n")

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
    def run(self, verbose:bool=None):
        if verbose is None:
            verbose = self.verbose

        print("casEN")
        if verbose:
            print("casEN verbose Activate")
    