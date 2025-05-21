import utils
import time
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import re
import pandas as pd
import json

class CasEN():

    def __init__(self, path:str="", data:str=None, allowed_grf:bool=False, remove_casEN_MISC:bool=True, archiving:bool=False, unique_corpus_file:bool=True, corpus_folder:str="", casEN_result:str="", make_excel:bool=False, timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.path = path
        self.data = data
        self.allowed_grf = allowed_grf
        self.remove_casEN_MISC = remove_casEN_MISC
        self.archiving = archiving
        self.corpus_folder = corpus_folder
        self.casEN_result = casEN_result
        self.timer_option = timer_option
        self.log_option = log_option
        self.verbose = verbose
        self.log_path = log_path
        self.make_excel = make_excel
        self.unique_corpus_file = unique_corpus_file

        self.log_location = Path(log_path) if log_path else Path("Logs/log.txt")

        # if the file is given then load it
        if isinstance(self.data, str):
            self.data = utils.load_data(data=self.data,timer_option=self.timer_option, log_option=self.log_option, verbose=self.verbose)


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

    # --------------------------- METHODS ----------------------- #
    @chrono
    def execute_casEN(self):
        get_ipython().run_line_magic('run', str(self.path))

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
            casEN_result_folder = self.casEN_result

        all_entities = []

        if self.unique_corpus_file:
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
    def make_df(self, verbose:bool=False) -> pd.DataFrame:

        entities_list = self.extract_casEN_entities()
        rows = []
        for entity in entities_list:
            ner_label =  self.find_ner_label(entity["tag"])
            if self.remove_casEN_MISC and ner_label == "MISC":
                continue
            idx = entity["file_id"]
            ner_text = entity["text"]
            context = self.get_ner_context(self.data.loc[idx, "desc"], ner_text, window=7)
            rows.append({
                "titles": self.data.loc[idx, "titles"],
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

    @chrono
    def casEN_optimisation(self, casEN_df: pd.DataFrame = None, allowed_grf: list = None, verbose: bool = None):
        """
            Parcourt toutes les lignes de `merge`, et pour celles où
            method == "casEN" ET dont la combinaison de graphes
            figure dans allowed_grf, remplace "casEN" par "casEN_opti".
            Ne supprime plus aucune ligne.
        """
        if casEN_df is None:
            casEN_df = self.casEN_df
        if allowed_grf is None:
            allowed_grf = self.allowed_grf
        if verbose is None:
            verbose = self.verbose
            
        with open(allowed_grf, 'r', encoding="utf-8") as f:
            allowed = json.load(f)

        def is_allowed(row):
            for combo in allowed:
                if all(row.get(col) == val for col, val in combo.items()):
                    return True
            return False

        if verbose:
            print("[opt] count par méthode avant :")
            print(casEN_df["method"].value_counts(dropna=False))

        def upgrade_method(row):
            if row["method"] == "casEN" and is_allowed(row):
                return "casEN_opti"
            else:
                return row["method"]

        casEN_df["method"] = casEN_df.apply(upgrade_method, axis=1)

        if verbose:
            print("[opt] count par méthode après :")
            print(casEN_df["method"].value_counts(dropna=False))


        self.casEN_df = casEN_df.reset_index(drop=True)
        return self.casEN_df

    @chrono
    def run(self, verbose:bool=None):
        if verbose is None:
            verbose = self.verbose

        # prepare folder
        utils.prepare_folder(folder_path=self.casEN_result, name="CasEN_Result",archiving=self.archiving, verbose=self.verbose)
        utils.prepare_folder(folder_path=self.corpus_folder, name="CasEN_Corpus", archiving=self.archiving, verbose=self.verbose)
        # make corpus
        utils.generate_casEN_file(df=self.data, corpus_path=self.corpus_folder, unique_file=self.unique_corpus_file, verbose=self.verbose)
        # run casEN
        self.execute_casEN()
        # make dataframe
        self.make_df()

        if self.allowed_grf:
            self.casEN_optimisation()

        return self.casEN_df
    