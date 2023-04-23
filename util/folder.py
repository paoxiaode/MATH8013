import os
import platform
import shutil
from .utils import mkdir
from datetime import datetime


def make_folder():
    # Make a folder to save all output
    BaseDir_neu = "test"
    if platform.system() == "Windows":
        # device_n="0"
        BaseDir0 = "../../../nn/%s" % (sBaseDir0)
        # BaseDir = '../../../nn/%s'%(sBaseDir)
    # else:
    # device_n="0"
    # BaseDir0 = sBaseDir0
    # BaseDir = sBaseDir
    # mkdir(BaseDir0)
    # BaseDir = '%s/%s' % (BaseDir0, example_folder)
    # mkdir(BaseDir)
    # BaseDir_a = '%s/%s' % (BaseDir, R['times'])
    # mkdir(BaseDir_a)
    # BaseDir_neu = '%s/%s' % (BaseDir_a, neu_ind_folder)
    mkdir(BaseDir_neu)
    subFolderName = "%s" % (datetime.now().strftime("%y%m%d%H%M%S"))
    # subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1))

    # subFolderName = '%s' % (
    #     int(np.absolute(np.random.normal([1]) * 100000)) // int(1))
    FolderName = "%s/%s/" % (BaseDir_neu, subFolderName)
    mkdir(FolderName)

    # mkdir('%smodel/'%(FolderName))
    # print(subFolderName)

    if not platform.system() == "Windows":
        shutil.copy(__file__, "%s%s" % (FolderName, os.path.basename(__file__)))
    return FolderName
