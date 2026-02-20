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

sns.set_style("whitegrid")
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
files = glob.glob("Alex_Lemonade_portal/**/*_filtered*.h5ad", recursive= True)
datasets = [sc.read_h5ad(f) for f in files]
ids = ["sample" + f.split("/")[-1].split("_")[0][-3:] for f in files]
merged_data = ann.concat(
    datasets,
    merge= "first", 
    keys= ids, 
    label= "sample", 
    index_unique= "-"
)
merged_data.obs["age"] = merged_data.obs["age"].astype(float)
merged_data.obs = merged_data.obs.drop(columns= "subdiagnosis")
merged_data.write_h5ad("data/merged_w_others.h5ad")

# %%
sc.pl.violin(
    merged_data,
    ["sum", "detected", "subsets_mito_percent"],
    multi_panel= True,
    save= "others_preproc.png"
)

# %%
sc.pl.scatter(
    merged_data,
    "sum",
    "detected",
    color= "subsets_mito_percent",
    color_map= "viridis",
    save= "others_sum_vs_detected.png"
)

# %%
merged_data.var["gene_symbol"] = merged_data.var["gene_symbol"].astype("str")
merged_data.var.loc[merged_data.var["gene_symbol"] == "nan", "gene_symbol"] = merged_data.var.loc[merged_data.var["gene_symbol"] == "nan"].index

# %%
sc.pl.highest_expr_genes(
    merged_data,
    gene_symbols= "gene_symbol",
    save= "others_merged.png"
)

# %%

sc.pp.highly_variable_genes(
    merged_data,
    n_top_genes= 2000,
    flavor= "seurat_v3",
    layer= analysis_layer,
)

sc.pl.highly_variable_genes(
    merged_data,
    save= "others_merged.png",
)

sc.pp.normalize_total(
    merged_data,
    exclude_highly_expressed= True,
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

merged_data = merged_data[:, merged_data.var["highly_variable"]]

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

    def score(self, X, y= None, sample_weight= None):
        return skm.silhouette_score(
            X.obsm["X_pca"],
            labels= X.obs["leiden"]
        )

# %%
pca = ScPCA(
    mask= "highly_variable",
    layer= analysis_layer,
)
neighbors = ScNeighbors()
scleid = ScLeiden()
scscorer = ScScore()
workflow = pipe.make_pipeline(pca, neighbors, scleid, scscorer)
param_grid = {
    "scpca__n_comps": range(10, 30),
    "scneighbors__n_neighbors": range(20, 50),
    "scleiden__resolution": np.linspace(0.1, 2, 10) 
}
X_train, X_test = skms.train_test_split(
    merged_data,
    test_size= 0.2,
    random_state= 0,
)


# %%
workflow.fit(merged_data)
workflow.score(merged_data)

# %%

grids = skms.HalvingGridSearchCV(
    workflow,
    param_grid= param_grid,
    min_resources= X_train.n_obs // 3 ** 4 ,
    n_jobs= -1,
) 

with joblib.parallel_backend("loky"):
    grids.fit(X_train)


# %%
grids.best_params_

# %%
with open("pickles/gridsearch_others_1000", "wb") as f:
    pickle.dump(grids, f)

# %%
fig, axs = plt.subplots(3)
for ax, param in zip(axs, param_grid.keys()):
    sns.lineplot(x= grids.cv_results_["param_" + param], y= grids.cv_results_["mean_test_score"], ax= ax)
    ax.set_title(param)
fig.tight_layout()
fig.savefig("figures/others_params_1000.pdf")

# %%
sc.pp.pca(
    merged_data,
    mask_var= "highly_variable",
    n_comps= grids.best_params_["scpca__n_comps"],
    layer= analysis_layer,
)

sc.pl.pca_variance_ratio(
    merged_data,
    log= True,
    save= "others_merged.png"
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
sc.pl.umap(
    merged_data,
    color= [
        "CDH1",
        "leiden", 
        "tissue_location",
        "disease_timing",
    ],
    gene_symbols= "gene_symbol",
    layer= analysis_layer,
    cmap= "inferno",
    palette= cc.glasbey_category10,
    save= "others_merged.png",
    vmax= 4,
    ncols= 1,
)

# %%
