import time
from datetime import datetime
from pathlib import Path
import pandas as pd


def load_data(data:str, verbose:bool=False) -> pd.DataFrame:
    print("load data")
    excel_path = Path(data)
    if not excel_path.is_file():
        raise FileNotFoundError(f"The Excel file doesn't exist : {excel_path}")
    df = pd.read_excel(excel_path)
    df["desc"] = df["desc"].fillna("") # clean the empty description
    return df



































# def log_step(step: str, duration: float, log_location: Path):
#     timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
#     if log_location.is_dir():
#         target = log_location / "log.txt"
#     else:
#         target = log_location
#     target.parent.mkdir(parents=True, exist_ok=True)
#     with open(target, "a", encoding="utf-8") as f:
#         f.write(f"{timestamp} - [{step}] finished in {duration:.2f}s\n")

# def chrono(timer_attr: str = "timer_option",
#            log_attr: str = "log_option",
#            log_method: str = "log"):

#     def decorator(func):
#         def wrapper(self, *args, **kwargs):
#             use_timer = getattr(self, timer_attr, False)
#             use_log   = getattr(self, log_attr,    False)
#             if use_timer or use_log:
#                 start = time.time()
#             result = func(self, *args, **kwargs)
#             if use_timer or use_log:
#                 duration = time.time() - start
#                 if use_timer:
#                     print(f"{func.__name__}: {duration:.2f}s")
#                 if use_log:
#                     # appelle self.log_step ou self.log selon votre nommage
#                     getattr(self, log_method)(func.__name__, duration)
#             return result
#         return wrapper
#     return decorator
