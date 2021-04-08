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

train = 0
plott = 0
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# GENE EXPRESSION DATA
# Preprocessing and register the data
adata = pbmc3k_gex.copy()
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
    gexmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/gex3kpbmc")
else:
    gexmodel = scvi.model.SCVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/gex3kpbmc/", adata,
                                    use_gpu=False)

if plott:
    # Get Posterior Outputs
    latent = gexmodel.get_latent_representation()
    adata.obsm["X_scVI"] = latent
    adata.layers["scvi_normalized"] = gexmodel.get_normalized_expression(library_size=10e4)

    # Scanpy Plotting
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata, min_dist=0.3)
    sc.tl.leiden(adata, key_added="leiden_scVI", resolution=1.0)
    sc.pl.umap(adata, color=["leiden_scVI"], frameon=False,)
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
# CHROMATIN ACCESSIBILITY DATA
# Preprocessing and register the data
adata2 = pbmc3k_chac
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
    ChAcmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/chac3kpbmc")
else:
    ChAcmodel = scvi.model.PEAKVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/chac3kpbmc/", adata2,
                                       use_gpu=False)

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
adata4 = pbmc3k.copy()
adata4.var_names_make_unique()
sc.pp.filter_genes(adata4, min_counts=3)
sc.pp.filter_cells(adata4, min_counts=3)
adata4.layers["counts"] = adata4.X.copy()
scvi.data.setup_anndata(adata4, layer="counts")

# Model and Training
if train == 1:
    RNATACmodel = scvi.model.SCPEAKVI(adata4, n_genes=36600, n_regions=130440)
    RNATACmodel.train(max_epochs=100)

    # Saving Gex Model
    RNATACmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3kpbmc")
else:
    RNATACmodel = scvi.model.SCPEAKVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3kpbmc/",
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
# #########################################################################################################
# #########################################################################################################
# #########################################################################################################

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

train = 1
if train == 1:
    TFmodel = scvi.model.TFVI(adata3, M=M)
    TFmodel.train(max_epochs=100)
    # Saving Gex Model
    TFmodel.save("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3kpbmc")
else:
    TFmodel = scvi.model.TFVI.load("/Users/mgabitto/Desktop/Projects/scvi-tools/data/models/RNATAC3kpbmc/", adata3,
                                   use_gpu=False)

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################
