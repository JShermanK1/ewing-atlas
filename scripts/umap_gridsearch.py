# %%
import scanpy as sc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
import colorcet as cc
import sklearn as sk
import sklearn.decomposition as decomp
import sklearn.pipeline as pipe
import sklearn.neighbors as nbr
import sklearn.base as skbase
import sklearn.model_selection as skms
import pickle
import os
import joblib
import glob
import anndata as ann

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
files = glob.glob("Alex_Lemonade_portal/**/*_filtered*.h5ad")
datasets = [sc.read_h5ad(f) for f in files]
ids = ["sample" + f.split("/")[-1].split("_")[0][-3:] for f in files]
merged_data = ann.concat(
    datasets,
    merge= "first", 
    keys= ids, 
    label= "sample", 
    index_unique= "-"
)

# %%
sc.pl.violin(
    merged_data,
    ["sum", "detected", "subsets_mito_percent"],
    multi_panel= True,
    save= "_preproc.png"
)

# %%
sns.set_style("whitegrid")
sc.pl.scatter(
    merged_data,
    "sum",
    "detected",
    color= "subsets_mito_percent",
    color_map= "viridis",
    save= "_sum_vs_detected.png"
)

# %%
merged_data.var["gene_symbol"] = merged_data.var["gene_symbol"].astype("str")
merged_data.var.loc[merged_data.var["gene_symbol"] == "nan", "gene_symbol"] = merged_data.var.loc[merged_data.var["gene_symbol"] == "nan"].index

# %%
sc.pl.highest_expr_genes(
    merged_data,
    gene_symbols= "gene_symbol",
    save= "_merged.png"
)

# %%
sc.pp.normalize_total(
    merged_data,
    exclude_highly_expressed= True,
    key_added= "norm_factor",
    layer= "spliced",
)

sc.pp.log1p(
    merged_data,
)

sc.pp.highly_variable_genes(
    merged_data,
    n_top_genes= 2000,
    flavor= "seurat_v3"
)

sc.pl.highly_variable_genes(
    merged_data,
    save= "_merged.png"
)

sc.pp.scale(
    merged_data,
)

# %%
class ScPCA(skbase.TransformerMixin, skbase.BaseEstimator):
    def __init__(self, layer= None, n_comps= None, mask= None):
        self.layer = layer
        self.n_comps = n_comps
        self.mask = mask

    def fit(self, X, y= None):
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

    def fit(self, X, y= None):
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

    def fit(self, X, y= None):
        return self

    def transform(self, X):
        sc.tl.leiden(
            X,
            resolution= self.resolution,
            flavor= "igraph",
        )
        return X

class ScScore(skbase.TransformerMixin, skbase.BaseEstimator):

    def fit(self, X, y= None):
        return self

    def score(estimator, X, y= None, sample_weight= None):
        return skm.silhouette_score(
            X.obsp["distances"],
            metric= "precomputed",
            labels= X.obs["leiden"]
        )

# %%
pca = ScPCA(mask= "highly_variable")
neighbors = ScNeighbors()
scleid = ScLeiden()
scscorer = ScScore()
workflow = pipe.make_pipeline(pca, neighbors, scleid, scscorer)
param_grid = {
    "scpca__n_comps": range(20, 40),
    "scneighbors__n_neighbors": range(40, 60),
    "scleiden__resolution": np.linspace(0.1, 2, 10) 
}
X_train, X_test = skms.train_test_split(
    merged_data,
    test_size= 0.2,
    random_state= 0,
)
kfold = skms.KFold(
    shuffle= True,
    random_state= 0,
)


# %%
workflow.fit(merged_data)
workflow.score(merged_data)

# %%

grids = skms.GridSearchCV(
    workflow,
    param_grid= param_grid,
    cv= kfold,
    return_train_score= True,
    pre_dispatch= 48
) 

with joblib.parallel_backend("loky"):
    grids.fit(X_train)


# %%
grids.best_params_

# %%
with open("pickles/gridsearch", "wb") as f:
    pickle.dump(grids, f)

# %%
fig, axs = plt.subplots(3)
for ax, param in zip(axs, param_grid.keys()):
    sns.lineplot(x= grids.cv_results_["param_" + param], y= grids.cv_results_["mean_test_score"], ax= ax)
    ax.set_title(param)
fig.tight_layout()
fig.savefig("figures/params.pdf")

# %%
sc.pp.pca(
    merged_data,
    layer= "norm_scaled_genes",
    mask_var= "highly_variable",
    n_comps= grids.best_params_["scpca__n_comps"]
)

sc.pl.pca_variance_ratio(
    merged_data,
    log= True,
    save= "_merged.png"
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
skm.silhouette_score(
    merged_data.obsp["distances"],
    metric= "precomputed",
    labels= merged_data.obs["leiden"]
)

# %%
sc.pl.umap(
    merged_data,
    color= [
        "CDH1",
        "leiden", 
    ],
    gene_symbols= "gene_symbol",
    cmap= "viridis",
    palette= cc.glasbey_category10,
    save= "_merged.png",
)

# %%