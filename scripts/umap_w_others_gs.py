# %%
import glob
import os
import pickle

import anndata as ann
import colorcet as cc
import joblib
import numpy as np
import scanpy as sc
import seaborn as sns
import sklearn.base as skbase
import sklearn.metrics as skm
import sklearn.model_selection as skms
import sklearn.pipeline as pipe
from matplotlib import pyplot as plt
from sklearn.experimental import enable_halving_search_cv
import argparse
import pathlib

sns.set_style("whitegrid")

apr = argparse.ArgumentParser(
    description= "Gridsearch for clustering arugments that optimize silhouette score",
)
apr.add_argument(
    "--anndata", "-a",
    type= pathlib.Path,
    required= True,
    help= "input annotated dataframe"
)
apr.add_argument(
    "--prefix", "-p",
    type= str,
    help= "prefix for outputs",
)
apr.add_argument(
    "--iterations", "-i",
    type= int,
    default= 3,
)
apr.add_argument(
    "--transpose", "-t",
    action= "store_true",
    help= "transpose anndata before gridsearch",
)

args = apr.parse_args()
if args.prefix:
    args.prefix += "-"

analysis_layer = None #None == "X"
# %%
os.makedirs(
    "figures",
    exist_ok= True,
)
os.makedirs(
    "pickles",
    exist_ok= True,
)
os.makedirs(
    "data",
    exist_ok= True,
)

# %%
merged_data = sc.read_h5ad(args.anndata)
# %%

sc.pp.normalize_total(
    merged_data,
    exclude_highly_expressed= False,
    key_added= "norm_factor",
    layer= analysis_layer,
)

sc.pp.log1p(
    merged_data,
    layer= analysis_layer,
)

sc.pp.scale(
    merged_data,
    layer= analysis_layer,
)

merged_data = merged_data[:, merged_data.var["highly_variable"]].copy()
if args.transpose:
    merged_data = merged_data.T

# %%
class ScPCA(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, layer= None, n_comps= None, mask= None):
        self.layer = layer
        self.n_comps = n_comps
        self.mask = mask

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self

    def transform(self, X):
        sc.pp.pca(
            X,
            n_comps= self.n_comps,
            mask_var= self.mask,
            layer= self.layer,
        )
        return X

class ScNeighbors(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, n_neighbors= 15, n_pcs= None):
        self.n_neighbors = n_neighbors
        self.n_pcs = n_pcs

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self
        
    def transform(self, X):
        sc.pp.neighbors(
            X,
            n_neighbors= self.n_neighbors,
            n_pcs= self.n_pcs,
        )
        return X
    
class ScLeiden(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, resolution= 1):
        self.resolution = resolution

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self

    def transform(self, X):
        sc.tl.leiden(
            X,
            resolution= self.resolution,
            flavor= "igraph",
        )
        return X

class ScScore(skbase.TransformerMixin, skbase.BaseEstimator):
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        return tags

    def fit(self, X, y= None):
        self._is_fitted = True
        return self

    def score(self, X, y= None, sample_weight= None):
        return skm.silhouette_score(
            X.obsm["X_pca"],
            labels= X.obs["leiden"]
        )

# %%
pca = ScPCA(
    layer= analysis_layer,
)
neighbors = ScNeighbors()
scleid = ScLeiden()
scscorer = ScScore()
workflow = pipe.make_pipeline(pca, neighbors, scleid, scscorer)
param_grid = {
    "scpca__n_comps": range(10, 30),
    "scneighbors__n_neighbors": range(20, 50),
    "scleiden__resolution": np.linspace(0.1, 1, 20) 
}
X_train, X_test = skms.train_test_split(
    merged_data,
    test_size= 0.2,
    random_state= 0,
)


# %%
workflow.fit(merged_data)
workflow.__sklearn_is_fitted__
workflow.score(merged_data)

# %%

grids = skms.HalvingGridSearchCV(
    workflow,
    param_grid= param_grid,
    min_resources= X_train.n_obs // 3 ** (args.iterations - 1),
    n_jobs= -1,
) 

with joblib.parallel_backend("loky"):
    grids.fit(X_train)


# %%
grids.best_params_

# %%
with open(f"pickles/{args.prefix}gridsearch_1000", "wb") as f:
    pickle.dump(grids, f)

# %%
fig, axs = plt.subplots(3)
for ax, param in zip(axs, param_grid.keys()):
    sns.lineplot(x= grids.cv_results_["param_" + param], y= grids.cv_results_["mean_test_score"], ax= ax)
    ax.set_title(param)
fig.tight_layout()
fig.savefig(f"figures/{args.prefix}params_1000.pdf")

# %%
sc.pp.pca(
    merged_data,
    n_comps= grids.best_params_["scpca__n_comps"],
    layer= analysis_layer,
)

sc.pl.pca_variance_ratio(
    merged_data,
    log= True,
    save= f"{args.prefix}merged.png"
)

# %%

# %%
sc.pp.neighbors(
    merged_data,
    n_neighbors= grids.best_params_["scneighbors__n_neighbors"],
)
sc.tl.umap(
    merged_data,
)
sc.tl.leiden(
    merged_data,
    resolution= grids.best_params_["scleiden__resolution"]
)
print(skm.silhouette_score(
    merged_data.obsm["X_pca"],
    labels= merged_data.obs["leiden"]
))

# %%


# %%
