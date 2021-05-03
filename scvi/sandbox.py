import scanpy as sc
import scvi
import numpy as np
import anndata as ad


# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# Load all scRNA-scATAC data
pbmc3k = sc.read_10x_h5(
    '/Users/mgabitto/Desktop/Projects/scvi-tools/data/PBMC3k_granulocyte_sorted_3k_filtered_feature_bc_matrix.h5',
    gex_only=False)
pbmc10k = sc.read_10x_h5(
    '/Users/mgabitto/Desktop/Projects/scvi-tools/data/PBMC10k_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5',
    gex_only=False)

pbmc3k_gex = pbmc3k[:, 0:36601]
pbmc3k_chac = pbmc3k[:, 36601:193209]

pbmc10k_gex = pbmc10k[:, 0:36601]
pbmc10k_chac = pbmc10k[:, 36601:144979]

data3k = False
train = 0
plott = 1
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# GENE EXPRESSION DATA
# Preprocessing and register the data
if data3k:
    adata = pbmc3k_gex.copy()
else:
    adata = pbmc10k_gex.copy()

adata.var_names_make_unique()
sc.pp.filter_genes(adata, min_counts=3)
sc.pp.filter_cells(adata, min_counts=3)
adata.layers["counts"] = adata.X.copy()

scvi.data.setup_anndata(adata, layer="counts")

# Model and Training
if train == 1:
    gexmodel = scvi.model.SCVI(adata=adata)
    gexmodel.train(max_epochs=100)

    # Saving Gex Model
    if data3k:
        gexmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/gex3kpbmc")
    else:
        gexmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/gex10kpbmc")
else:
    if data3k:
        gexmodel = scvi.model.SCVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/gex3kpbmc/", adata,
                                        use_gpu=False)
    else:
        gexmodel = scvi.model.SCVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/gex10kpbmc/", adata,
                                        use_gpu=False)

if plott:
    # Get Posterior Outputs
    latent = gexmodel.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    adata.layers["scvi_normalized"] = gexmodel.get_normalized_expression(library_size=10e4)

    # Scanpy Plotting
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, key_added="leiden_scVI", resolution=2.0)
    sc.pl.umap(adata, color=["leiden_scVI"], frameon=False,)

    cell_markers = ["leiden_scVI",
                    "CD19", "CD24", "PAX5",      # B         Clusters: 17, 10, 23, 27
                    "JCHAIN", "MZB1", "XBP1",    # Plasma    Clusters: -
                    "RUNX2", "IRF8", "TCF4",     # pDC       Clusters: 24
                    "CLEC9A", "XCR1", "LAMP3",   # DC1       Clusters: -
                    "CD1E","CLEC10A","PKIB",     # DC2       Clusters: 21
                    "CD4",                       # T CD4     Clusters: 1,14,15,7,3,9,27
                    "FOXP3",                     # T Reg     Clusters: 15
                    "CD8A", "CD8B",              # T CD8     Clusters: 0, 16, 5, 12
                    "CCL5","KLRB1","GZMA",       # NK/NKT    Clusters: inconclusive.
                    "HBA1", "HBB",               # Eryt      Clusters: -
                    "S100A8", "TYROBP",          # Neutrop   Clusters: inconclusive.
                    "C1QA", "C1QC", "MARCO"]      # MacroPha Clusters:inconclusive.
    sc.pl.umap(adata, color=cell_markers, legend_loc="on data", frameon=False, ncols=3,  wspace=0.1)

    # DE Genes
    de_df = gexmodel.differential_expression(groupby="leiden_scVI", delta=0.5, batch_correction=False)
    filtered_rna = {}
    cats = adata.obs.leiden_scVI.cat.categories
    for i, c in enumerate(cats):
        cid = "{} vs Rest".format(c)
        cell_type_df = de_df.loc[de_df.comparison == cid]
        cell_type_df = cell_type_df.sort_values("lfc_median", ascending=False)
        cell_type_df = cell_type_df[cell_type_df.lfc_median > 0]
        data_rna = cell_type_df
        data_rna = data_rna[data_rna["bayes_factor"] > 2]
        data_rna = data_rna[data_rna["non_zeros_proportion1"] > 0.05]
        filtered_rna[c] = data_rna.index.tolist()[:2]
    sc.tl.dendrogram(adata, groupby="leiden_scVI", use_rep="X_scVI")
    sc.pl.dotplot(adata, filtered_rna, groupby="leiden_scVI", dendrogram=True, standard_scale="var", swap_axes=True)


# #########################################################################################################
# #########################################################################################################
# #########################################################################################################



# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# CHROMATIN ACCESSIBILITY DATA
# Preprocessing and register the data
if data3k:
    adata2 = pbmc3k_chac
else:
    adata2 = pbmc10k_chac
adata2.var_names_make_unique()
sc.pp.filter_genes(adata2, min_counts=3)
sc.pp.filter_cells(adata2, min_counts=3)
adata2.layers["counts"] = adata2.X.copy()
scvi.data.setup_anndata(adata2, layer="counts")

# Model and Training
if train == 1:
    ChAcmodel = scvi.model.PEAKVI(adata2)
    ChAcmodel.train(max_epochs=100)

    # Saving Gex Model
    if data3k:
        ChAcmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/chac3kpbmc")
    else:
        ChAcmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/chac10kpbmc")
else:
    if data3k:
        ChAcmodel = scvi.model.PEAKVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/chac3kpbmc/",
                                           adata2, use_gpu=False)
    else:
        ChAcmodel = scvi.model.PEAKVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/chac10kpbmc/",
                                           adata2, use_gpu=False)

if plott:
    # Get Posterior Outputs
    latent = ChAcmodel.get_latent_representation()
    adata2.obsm["X_scVI"] = latent

    # Scanpy Plotting
    sc.pp.neighbors(adata2, use_rep="X_scVI")
    sc.tl.umap(adata2, min_dist=0.3)
    sc.tl.leiden(adata2, key_added="leiden_peakVI", resolution=1.0)
    sc.pl.umap(adata2, color=["leiden_peakVI"], frameon=False,)
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# RNA/ATAC
if data3k:
    adata4 = pbmc3k.copy()
else:
    adata4 = pbmc10k.copy()
adata4.var_names_make_unique()
sc.pp.filter_genes(adata4, min_counts=3)
sc.pp.filter_cells(adata4, min_counts=3)
adata4.layers["counts"] = adata4.X.copy()
scvi.data.setup_anndata(adata4, layer="counts")

# Model and Training
if train == 1:
    if data3k:
        RNATACmodel = scvi.model.SCPEAKVITWO(adata4, n_genes=36600, n_regions=130440)
    else:
        RNATACmodel = scvi.model.SCPEAKVITWO(adata4, n_genes=26393, n_regions=107394)
    RNATACmodel.train(max_epochs=5)

    # Saving Gex Model
    if data3k:
        RNATACmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3-3kpbmc")
    else:
        RNATACmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3-10kpbmc")
else:
    if data3k:
        RNATACmodel = scvi.model.SCPEAKVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3-3kpbmc/",
                                               adata4, use_gpu=False)
    else:
        RNATACmodel = scvi.model.SCPEAKVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3-10kpbmc/",
                                               adata4, use_gpu=False)

if plott:
    # Get Posterior Outputs
    latent = RNATACmodel.get_latent_representation()
    adata4.obsm["X_scVI"] = latent

    # Scanpy Plotting
    sc.pp.neighbors(adata4, use_rep="X_scVI")
    sc.tl.umap(adata4, min_dist=0.3)
    sc.tl.leiden(adata4, key_added="leiden_peakVI", resolution=1.0)
    sc.pl.umap(adata4, color=["leiden_peakVI"], frameon=False,)

    latents = RNATACmodel.get_latents()

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
quit()
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# TFVI
adata3 = pbmc3k_chac.copy()
adata3.var_names_make_unique()
sc.pp.filter_genes(adata3, min_counts=3)
sc.pp.filter_cells(adata3, min_counts=3)
adata3.layers["counts"] = adata3.X.copy()

# TFBS = [cells x TFs]
adata3.obsm["tf_counts"] = np.ones((adata3.shape[0], 10))

# M = [Peaks x TFs]
M = np.ones((adata3.shape[1], 10))

scvi.data.setup_anndata(adata3, layer="counts", protein_expression_obsm_key="tf_counts")

if train == 1:
    TFmodel = scvi.model.TFVI(adata3, M=M)
    TFmodel.train(max_epochs=100)
    # Saving Gex Model
    TFmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/TF3kpbmc")
else:
    TFmodel = scvi.model.TFVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/TF3kpbmc/", adata3,
                                   use_gpu=False)

if plott:
    # Get Posterior Outputs
    latent = TFmodel.get_latent_representation()
    adata3.obsm["X_scVI"] = latent

    # Scanpy Plotting
    sc.pp.neighbors(adata3, use_rep="X_scVI")
    sc.tl.umap(adata3, min_dist=0.3)
    sc.tl.leiden(adata3, key_added="leiden_peakVI", resolution=1.0)
    sc.pl.umap(adata3, color=["leiden_peakVI"], frameon=False,)
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
