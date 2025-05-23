from spacy_wrapper import SpaCy
from casen import CasEN
import pandas as pd
from pathlib import Path
import utils
import time
from datetime import datetime
import json

class Pipeline:

    def __init__(self, spaCy:SpaCy=None, casEN:CasEN=None, data:str="", casEN_enrichment:bool=True, grf:str="" ,correction_path:str=None, pipeline_result:str="", remove_duplicate_rows:bool=False, timer_option:bool=False, log_option:bool=False, log_path:str="", verbose:bool=False):
        self.spaCy = spaCy
        self.casEN = casEN

        self.data = data
        self.casEN_enrichment = casEN_enrichment
        self.grf = grf
        self.correction_path = correction_path
        self.pipeline_result = pipeline_result
        self.remove_duplicate_rows = remove_duplicate_rows
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
    def merge_enrichment(self,df:pd.DataFrame=None, verbose:bool=None) -> pd.DataFrame:
        """ Performs cross enrichment: if both spaCy and casEN detect the same entity with different labels, prefer casEN’s label for higher accuracy """
        verbose = verbose or self.verbose

        # filter spaCy and casEN
        spaCy_df = df[df["method"] == "spaCy"]
        casEN_df = df[df["method"] == "casEN"]

        merged = pd.merge(spaCy_df, casEN_df, on=["NER", "file_id"], suffixes=("_spacy","_casen"))

        conflicts = merged[merged["NER_label_spacy"] != merged["NER_label_casen"]]

        if verbose:
            print(f"[merge_enrichment] {len(conflicts)} conflicting entities found (spaCy vs casEN)")

        def is_allowed(row):
            with open(self.grf, 'r', encoding="utf-8") as f:
                allowed = json.load(f)  

            for combo in allowed:
                if (
                    row.get("main_graph_casen") == combo.get("main_graph") and
                    row.get("second_graph_casen") == combo.get("second_graph") and
                    row.get("third_graph_casen") == combo.get("third_graph")
                ):
                    return True
            return False

        with open("C:\\Users\\valen\\Documents\\Informatique-L3\\Stage\\Stage\\name.json", 'r', encoding="utf-8") as f:
            name = json.load(f)

        name_list = name[0].get("NER")

        new_rows = []
        for _, row in conflicts.iterrows():
            #if is_allowed(row): # if we want to check graphes
            if row["NER_label_casen"] == "PER" and row["NER"] not in name_list:
                new_rows.append({
                    "titles": row["titles_spacy"],
                    "NER": row["NER"],
                    "NER_label": row["NER_label_casen"],
                    "desc": row["desc_spacy"],           
                    "method": "cross-enrichment",
                    "main_graph" : row["main_graph_casen"],
                    "second_graph" : row["second_graph_casen"],
                    "third_graph" : row["third_graph_casen"],
                    "file_id": row["file_id"]
                })
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df = df.drop_duplicates(subset=["titles", "NER", "NER_label", "desc", "method","main_graph", "second_graph", "third_graph", "file_id"])
            if verbose:
                print(f"[merge_enrichment] {len(new_rows)} added.")

        return df

    @chrono
    def merge_spacy_casEN(self, spaCy_df:pd.DataFrame=None, casEN_df:pd.DataFrame=None, verbose:bool=False) -> pd.DataFrame:
        """

        """
        if spaCy_df is None:
            spaCy_df = self.spaCy_df
        if casEN_df is None:
            casEN_df = self.casEN_df

        if verbose == True:
            df_non_vides = self.data[self.data["desc"] != ""]

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
        

        if self.casEN_enrichment:
            merge = self.merge_enrichment(df=merge, verbose=verbose)

        if self.remove_duplicate_rows:
            merge = merge.drop_duplicates(subset=["titles", "NER", "NER_label", "method", "main_graph", "second_graph", "third_graph"])
            if verbose:
                print("[merge] dropping duplicate rows")

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

        self.merge = merge[final_columns]

        if verbose:
            files_with_entities = set(merge["file_id"])
            total_with_desc = len(df_non_vides)
            desc_without_entity = total_with_desc - len(files_with_entities)
            print(f"[merge] description without entities : {desc_without_entity}")


        return self.merge

    @chrono
    def correct_excel(self, merge:pd.DataFrame=None, verbose:bool=None):

        if verbose is None:
            verbose = self.verbose

        if merge is None:
            merge = self.merge

        correction_df = pd.read_excel(self.correction_path)

        correction_df["key"] = correction_df[["NER", "NER_label", "hash"]].apply(tuple, axis=1)
        merge["key"] = merge[["NER", "NER_label","file_id"]].apply(tuple, axis=1)

        cols_to_copy = ["manual cat", "correct", "extent", "category"]

        correction_dict = correction_df.set_index("key")[cols_to_copy].to_dict(orient="index")

        for col in cols_to_copy:
            merge[col] = merge["key"].map(lambda k: correction_dict.get(k, {}).get(col, ""))

        merge.drop(columns=["key"], inplace=True)

        self.merge = merge

        return self.merge

    
    
    @chrono
    def run(self, spaCy:SpaCy=None, casEN:CasEN=None):

        # ----- INIT ----- #
        data_df = utils.load_data(data=self.data,timer_option=self.timer_option, log_option=self.log_option, verbose=self.verbose)

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
                path = "D:\\travail\\Stage\\Stage_NER\\CasEN_fr\\CasEN_fr.2.0\\CasEN.ipynb",
                data = data_df,
                trustable_grf = False,
                remove_casEN_MISC = True,
                archiving = True,
                unique_corpus_file = True,
                corpus_folder = "D:\\travail\\Stage\\Stage_NER\\Result\\Corpus",
                casEN_result = "D:\\travail\\Stage\\Stage_NER\\Result\\CasEN_Result\\Res_CasEN_Analyse_synthese_grf",
                make_excel = False,
                timer_option = True,
                log_option = True,
                log_path = "D:\\travail\\Stage\\Stage_NER\\Pipeline\Logs",
                verbose = True
            )
        else:
            casEN.data = data_df

        self.casEN = casEN

        # ------- RUN -------- #
        self.spaCy_df = spaCy.run()
        self.casEN_df = casEN.run()

        # ----- MERGE --- #
        self.merge_spacy_casEN()

        # ---- CORRECTION ---- #
        if self.correction_path != None:
            self.correct_excel()

        # --------- Generate Excel file  ------
        filename = f"Pipeline_enrichment_{self.casEN_enrichment}_casEN_opti_{True if self.casEN.allowed_grf else False}_duplicated_rows_{self.remove_duplicate_rows}"
        save = utils.save_dataframe(self.merge, filename, self.pipeline_result)

        return save
        

    