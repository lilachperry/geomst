
import os
from glob import glob
from subprocess import call
import numpy as np
from random import shuffle



dirname = '/mnt/data/messin/dev/python/fontClassifier/datasets/Capitals64/seperateChars/completeDataset'
dirname2 = '/mnt/data/noafish/geomst/datasets/fonts'
trd = os.path.join(dirname2, 'train')
ted = os.path.join(dirname2, 'test')

lst = os.listdir(dirname)
lst.sort()
shuffle(lst)

num = len(lst)
tr = np.round(0.9*num)

if not os.path.exists(trd):
    os.makedirs(trd)
    os.makedirs(ted)


for idx, cdir in enumerate(lst):
    cdirf = os.path.join(dirname, cdir)
    lst2 = os.listdir(cdirf)

    for idx2, fnm in enumerate(lst2):
        f = os.path.join(cdirf, fnm)
        if idx < tr:
            f2 = os.path.join(trd, fnm)
        else:
            f2 = os.path.join(ted, fnm)
        call(["ln", "-s", f, f2])



# #result = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], '*.jpg'))]
# result = [y for x in os.walk(dirname) for y in glob(os.path.join(x[0], '*.png'))]
# sz = result.__len__()
# tr = sz
# #tr = np.round(0.8 * sz)
# #vl = np.round(0.1 * sz)
# #ts = sz - tr - vl

# shuffle(result)

# for idx, f in enumerate(result):
#     print(f)
#     if not os.path.isdir(tra):
#         os.mkdir(tra)
#         os.mkdir(trb)
#     call(["ln", "-s", f, tra])
#     call(["ln", "-s", f, trb])
#     # if idx < tr:
#     #     if not os.path.isdir(tra):
#     #         os.mkdir(tra)
#     #         os.mkdir(trb)
#     #     call(["ln", "-s", f, tra])
#     #     call(["ln", "-s", f, trb])
#     # elif idx < tr + vl:
#     #     if not os.path.isdir(vla):
#     #         os.mkdir(vla)
#     #         os.mkdir(vlb)
#     #     call(["ln", "-s", f, vla])
#     #     call(["ln", "-s", f, vlb])
#     # else:
#     #     if not os.path.isdir(tsa):
#     #         os.mkdir(tsa)
#     #         os.mkdir(tsb)
#     #     call(["ln", "-s", f, tsa])
#     #     call(["ln", "-s", f, tsb])




