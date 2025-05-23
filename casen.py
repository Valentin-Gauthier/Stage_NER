from pathlib import Path
import pandas as pd

class CasEN:
    def __init__(self,data:str, run_casEN:bool=False, verbose:bool=False):
        self.data = data
        self.run_casEN = run_casEN
        self.verbose = verbose


    def load(self, folder:str) -> list:
        """Load CasEN result"""

        path = Path(folder)

        if not path.exists():
            raise FileNotFoundError(f"[load] The provided folder does not exist : {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"[load] The provided path is not a folder : {path}")
        
        casEN_results = list(path.glob("*.txt"))

        file_counts = len(casEN_results)
        if file_counts == 0:
            raise Exception(f"[load] No file(s) to load")
        else:
            if self.verbose:
                print(f"[load] {file_counts} file(s) loaded")
        
        return casEN_results


    def run(self) -> pd.DataFrame:
        """Execute all steps to make a DataFrame with CasEN"""

        if self.run_casEN:
            ...
        else:
            self.load(self.data)