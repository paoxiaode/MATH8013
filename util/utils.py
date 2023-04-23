import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
Leftp = 0.18
Bottomp = 0.18
Widthp = 0.88 - Leftp
Heightp = 0.9 - Bottomp
pos = [Leftp, Bottomp, Widthp, Heightp]


def func0(xx):
    y_sin = np.sin(xx) + 2 * np.sin(3 * xx) + 3 * np.sin(5 * xx)
    return y_sin


def get_y(xx, alpha=1):
    y_sin = func0(xx)
    if alpha == 0:
        return y_sin
    out_y = np.round(y_sin / alpha)
    out_y2 = out_y * alpha
    return out_y2


def mkdir(fn):  # Create a directory
    if not os.path.isdir(fn):
        os.mkdir(fn)


def save_fig(pltm, fntmp, fp=0, ax=0, isax=0, iseps=0, isShowPic=0):  # Save the figure
    if isax == 1:
        pltm.rc("xtick", labelsize=18)
        pltm.rc("ytick", labelsize=10)
        ax.set_position(pos, which="both")
    fnm = "%s.png" % (fntmp)
    pltm.savefig(fnm)
    if iseps:
        fnm = "%s.eps" % (fntmp)
        pltm.savefig(fnm, format="eps", dpi=600)
    if fp != 0:
        fp.savefig("%s.pdf" % (fntmp), bbox_inches="tight")
    if isShowPic == 1:
        pltm.show()
    elif isShowPic == -1:
        return
    else:
        pltm.close()
