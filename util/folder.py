import os
import platform
import shutil
from .utils import mkdir
from datetime import datetime


def make_folder(R):
    # Make a folder to save all output
    BaseDir_neu = f"""log/{R["log_dir"]}"""
    FolderName = f"""{BaseDir_neu}/result_hidden{R["hidden_dim"]}_act{R["ActFuc"]}_fre{R["ActFre"]}/"""
    mkdir(FolderName)
    if not platform.system() == "Windows":
        shutil.copy(__file__, "%s%s" % (FolderName, os.path.basename(__file__)))
    return FolderName
