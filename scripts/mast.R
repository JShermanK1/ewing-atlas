library(anndataR)
library(SingleCellExperiment)
library(MAST)
library(parallelly)

options(mc.cores = availableCores())

sce <- read_h5ad(
  "data/merged_w_others_tumor.h5ad",
  as = "SingleCellExperiment"
)

cd2 <- colSums(assay(sce, i = 4) > 0)
colData(sce)$detected <- scale(cd2)

colData(sce)$idents <- as.factor(
    gsub(
        " ",
        "_",
        as.character(colData(sce)$idents)
    )
)

colData(sce)$disease_timing <- as.factor(
    gsub(
        " ",
        "_",
        as.character(colData(sce)$disease_timing)
    )
)

sca <- SceToSingleCellAssay(sce)
zlm_mod <- zlm(
  ~ idents + (1 | sample_id) + detected,
  sca,
  exprs_values = "log",
  method = "glmer",
  ebayes = F,
  strictConvergence = F,
)

saveRDS(
  zlm_mod,
  file = "data/mast_out_tumor.rds"
)
