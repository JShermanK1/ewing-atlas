library(anndataR)
library(SingleCellExperiment)
library(MAST)
library(parallelly)

options(mc.cores = availableCores())

sce <- read_h5ad(
  "data/combicells.h5ad",
  as = "SingleCellExperiment"
)

cd2 <- colSums(assay(sce, i = 4) > 0)
colData(sce)$detected <- scale(cd2)

sca <- SceToSingleCellAssay(sce)
zlm_mod <- zlm(
  ~ idents + diagnosis + disease_timing,
  sca,
  exprs_values = "log",
)

saveRDS(
  zlm_mod,
  file = "data/mast_out_combi.rds"
)
