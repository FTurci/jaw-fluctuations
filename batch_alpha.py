from src import alphavols
import pickle
import numpy as np
import pandas as pd
from joblib import delayed, Parallel
import cProfile
data_path = "data"

f = open(f"{data_path}/jaw_images_mutant_with_alpha.pkl", "rb")
jaws = []

while True:
    try:
        load = (pickle.load(f))
        name  = load['name'].lower()
        if 'col11' in name or 'control' in name:
            jaws.append(load)
    except EOFError:
        break
f.close()

# for j in jaws[:2]:
def do(j, tqdmon=False):
    analysis = alphavols.AlphaAnalyser(j)
    analysis.compute_inner_voxels(tqdmon=tqdmon)
    # vols = np.array([alphavols.volume(analysis.alpha['points'][s]) for s in analysis.alpha['simplices']])
    analysis.dump()
    # for i,j in zip(analysis.num_voxels, vols):
        # print(i,j)
    # print (len(analysis.num_voxels), len(analysis.alpha['simplices']))


# cProfile.run("do(jaws[1],tqdmon=True)")

# do(jaws[1],tqdmon=True)
# res = 
Parallel(n_jobs=4)(delayed(do) (j) for j in jaws)