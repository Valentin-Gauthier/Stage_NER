from pathlib import Path
import pandas as pd
import shutil
import time
from datetime import datetime

 # ---------------------------- TOOLS ----------------------- #
# def log(self, step:str, duration:float):
#     timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

#     if self.log_location.is_dir(): 
#         log_file_path = self.log_location / "log.txt" # if the path is a folder -> add a filename
#     else:
#         log_file_path = self.log_location


#     log_file_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(log_file_path, 'a', encoding="utf-8") as f:
#         f.write(f"{timestamp} - Pipeline [{step}] finish in {duration:.2f} s.\n")

# def chrono(func):
#     def wrapper(self, *args, **kwargs):
#         if self.timer_option or self.log_option:
#             start = time.time()
#         result = func(self, *args, **kwargs)
#         if self.timer_option or self.log_option:
#             duration = time.time() - start
#             if self.timer_option:
#                 print(f"{func.__name__} in : {duration:.2f}s")
#             if self.log_option:
#                 self.log(func.__name__, duration)
#         return result
#     return wrapper

# ---------------------------- METHODS ----------------------- #

def load_data(data:str, timer_option:bool=False, log_option:bool=False, verbose:bool=False) -> pd.DataFrame:
    print("load data")
    excel_path = Path(data)
    if not excel_path.is_file():
        raise FileNotFoundError(f"The Excel file doesn't exist : {excel_path}")
    df = pd.read_excel(excel_path)
    df["desc"] = df["desc"].fillna("") # clean the empty description
    return df


def prepare_folder(folder_path:str="Default", name:str="", archiving:bool=True, timer_option:bool=False, log_option:bool=False, verbose:bool=False):
        folder = Path(folder_path)
        # check if the path is good
        if not folder.exists():
            raise FileNotFoundError(f"casEN corpus folder does not exist : {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Provided casEN path is not a folder : {folder}")
        if verbose:
            print(f"[prepare folder] folder : {folder}")


        # get all the files
        files = list(folder.iterdir())
        if not files:
            if verbose:
                print(f"[prepare folder] Folder is empty")

        if archiving and files:
            # move every files into Archives folder
            archiving_folder = Path("Archives")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target = archiving_folder / f"{name}_{timestamp}"
            target.mkdir()

            if verbose :
                print(f"[prepare folder] Archiving contents to : {target}")
            
            for file in files:
                try:
                    shutil.move(str(file), str(target / file.name))
                    if verbose:
                        print(f"[prepare folder] Moved: {file.name} → {target}")
                except Exception as e:
                    print(f"[prepare folder] Failed to move {file}: {e}")
        else: 
            for file in files:
                try:
                    if file.is_file() or file.is_symlink():
                        file.unlink()
                        if verbose:
                            print(f"[prepare folder] Deleted file : {file.name}")
                    elif file.is_dir():
                        shutil.rmtree(file)
                        if verbose:
                            print(f"[prepare folder] Deleted folder : {file.name}")
                except Exception as e:
                    print(f"[prepare folder] Failed to delete {file} : {e}")


def generate_casEN_file(df:pd.DataFrame=None, corpus_path:str="", unique_file: bool = False, timer_option:bool=False, log_option:bool=False, verbose: bool = False):

        if unique_file:
            # Case 1: Generate a single corpus.txt file
            missing_desc = (df["desc"] == "").sum()
            corpus_path = Path(corpus_path) / "corpus.txt"
            with open(corpus_path, 'w', encoding="utf-8") as f:
                for idx, row in df.iterrows():
                    f.write(f'<doc id="{idx}">')
                    f.write(str(row["desc"]))
                    f.write('</doc>\n')
            if verbose:
                print(f"[generate file(s)] Single corpus file generated: {corpus_path}")
                print(f"[generate file(s)] Missing description : {missing_desc}")
        else:
            # Case 2: Generate one file per row
            for idx, row in df.iterrows():
                doc_path = Path(corpus_path) / f"doc_{idx}.txt"
                with open(doc_path, 'w', encoding="utf-8") as f:
                    f.write(f'<doc id="{idx}">')
                    f.write(str(row["desc"]))
                    f.write('</doc>\n')
            if verbose:
                print(f"[generate file(s)] {len(df)} individual corpus files generated in {corpus_path}")
                print(f"[generate file(s)] Missing description : {missing_desc}")





























