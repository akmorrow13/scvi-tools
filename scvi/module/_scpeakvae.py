from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Normal, kl_divergence as kl

from scvi import _CONSTANTS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers, DecoderSCVI
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial


class Decoder(torch.nn.Module):
    """
    Decodes data from latent space of ``n_input`` dimensions ``n_output``dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers. If False (default),
        covairates will only be included in the input layer.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            activation_fn=torch.nn.LeakyReLU,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(n_hidden, n_output), torch.nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, *cat_list: int):
        x = self.output(self.px_decoder(z, *cat_list))
        return x


class SCPEAKVAE(BaseModuleClass):
    """
    Variational auto-encoder model for ATAC-seq data.

    This is an implementation of the peakVI model descibed in.

    Parameters
    ----------
    n_input_regions
        Number of input regions.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer. If `None`, defaults to square root
        of number of regions.
    n_latent
        Dimensionality of the latent space. If `None`, defaults to square root
        of `n_hidden`.
    n_layers_encoder
        Number of hidden layers used for encoder NN.
    n_layers_decoder
        Number of hidden layers used for decoder NN.
    dropout_rate
        Dropout rate for neural networks
    model_depth
        Model library size factors or not.
    region_factors
        Include region-specific factors in the model
    use_batch_norm
        One of the following

        * ``'encoder'`` - use batch normalization in the encoder only
        * ``'decoder'`` - use batch normalization in the decoder only
        * ``'none'`` - do not use batch normalization (default)
        * ``'both'`` - use batch normalization in both the encoder and decoder
    use_layer_norm
        One of the following

        * ``'encoder'`` - use layer normalization in the encoder only
        * ``'decoder'`` - use layer normalization in the decoder only
        * ``'none'`` - do not use layer normalization
        * ``'both'`` - use layer normalization in both the encoder and decoder (default)
    latent_distribution
        which latent distribution to use, options are

        * ``'normal'`` - Normal distribution (default)
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False (default),
        covairates will only be included in the input layer.

    """

    def __init__(
        self,
        # PEAKVI
        n_input_regions: int = 0,
        n_batch: int = 0,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = 10,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        model_depth: bool = True,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: str = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,
        # SCVI INPUTS
        n_input: int = 0,
        n_labels: int = 0,
        dispersion: str = "gene",
        gene_likelihood: str = "zinb",
        log_variational: bool = True,
        use_observed_lib_size: bool = True,
    ):
        super().__init__()

        # INIT PARAMS PEAKVI
        self.n_input_regions = n_input_regions
        self.n_input = n_input
        self.n_hidden = (int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden)
        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.model_depth = model_depth
        self.dropout_rate = dropout_rate
        self.latent_distribution = latent_distribution
        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.deeply_inject_covariates = deeply_inject_covariates
        self.encode_covariates = encode_covariates

        cat_list = ([n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else [])

        # INIT PARAMS SCVI
        self.dispersion = dispersion
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.use_observed_lib_size = use_observed_lib_size

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was {}.format(self.dispersion)")

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # MODIFICATION ENCODER ENCODES ENTIRE VECTOR OF GENES + REGIONS
        n_input_encoder = self.n_input + self.n_input_regions + n_continuous_cov * encode_covariates
        encoder_cat_list = cat_list if encode_covariates else None

        self.z_encoder = Encoder(n_input=n_input_encoder, n_layers=self.n_layers_encoder, n_output=self.n_latent,
                                 n_hidden=self.n_hidden, n_cat_list=encoder_cat_list, dropout_rate=self.dropout_rate,
                                 activation_fn=torch.nn.LeakyReLU, distribution=self.latent_distribution, var_eps=0,
                                 use_batch_norm=self.use_batch_norm_encoder, use_layer_norm=self.use_layer_norm_encoder)

        # SCVI DECODER
        n_input_decoder = n_latent + n_continuous_cov
        self.decoder = DecoderSCVI(n_input_decoder, n_input, n_cat_list=cat_list, n_layers=n_layers_decoder,
                                   n_hidden=self.n_hidden, inject_covariates=deeply_inject_covariates,
                                   use_batch_norm=use_batch_norm_decoder, use_layer_norm=use_layer_norm_decoder,)

        # PEAKVI DECODER
        self.z_decoder = Decoder(n_input=self.n_latent + self.n_continuous_cov, n_output=n_input_regions,
                                 n_hidden=self.n_hidden, n_cat_list=cat_list, n_layers=self.n_layers_decoder,
                                 use_batch_norm=self.use_batch_norm_decoder, use_layer_norm=self.use_layer_norm_decoder,
                                 deep_inject_covariates=self.deeply_inject_covariates,)

        # PEAK VI ADDITIONAL ENCODERS
        self.d_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.d_encoder = Decoder(n_input=n_input_regions, n_output=1, n_hidden=self.n_hidden,
                                     n_cat_list=encoder_cat_list, n_layers=self.n_layers_encoder,)
        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

        # SCVI ADDITIONAL ENCODERS
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(n_input_encoder - n_input_regions, 1, n_layers=1, n_cat_list=encoder_cat_list,
                                 n_hidden=self.n_hidden, dropout_rate=dropout_rate,
                                 inject_covariates=deeply_inject_covariates,
                                 use_batch_norm=use_batch_norm_encoder, use_layer_norm=use_layer_norm_encoder, )

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        cont_covs = tensors.get(_CONSTANTS.CONT_COVS_KEY)
        cat_covs = tensors.get(_CONSTANTS.CAT_COVS_KEY)
        input_dict = dict(x=x, batch_index=batch_index, cont_covs=cont_covs, cat_covs=cat_covs,)
        return input_dict

    @auto_move_data
    def inference(self, x, batch_index, cont_covs, cat_covs, n_samples=1,) -> Dict[str, torch.Tensor]:

        # Get Data and Additional Covs
        x_rna = x[:, :self.n_input]
        x_chr = x[:, self.n_input:]
        if self.use_observed_lib_size:
            library = torch.log(x_rna.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_rna = torch.log(1 + x_rna)

        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat((x, cont_covs), dim=-1)
            encoder_rna = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_chr = torch.cat((x_chr, cont_covs), dim=-1)
        else:
            encoder_input = x
            encoder_rna = x_rna
            encoder_chr = x_chr

        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # Z Encoders
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        #   d Encoder PeakVI
        d = (self.d_encoder(encoder_chr, batch_index, *categorical_input) if self.model_depth else 1)
        #   l Encoder SCVI
        ql_m, ql_v, library_encoded = self.l_encoder(encoder_rna, batch_index, *categorical_input)

        if not self.use_observed_lib_size:
            library = library_encoded

        # ReFormat Outputs
        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))

            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

            ql_m = ql_m.unsqueeze(0).expand((n_samples, ql_m.size(0), ql_m.size(1)))
            ql_v = ql_v.unsqueeze(0).expand((n_samples, ql_v.size(0), ql_v.size(1)))
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = Normal(ql_m, ql_v.sqrt()).sample()

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, d=d, library=library)
        return outputs

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz_m"]
        library = inference_outputs["library"]
        y = tensors[_CONSTANTS.LABELS_KEY]

        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        cont_key = _CONSTANTS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = _CONSTANTS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        input_dict = {"z": z, "qz_m": qz_m, "batch_index": batch_index, "cont_covs": cont_covs, "cat_covs": cat_covs,
                      "library": library, "y": y}

        return input_dict

    @auto_move_data
    def generative(self, z, qz_m, batch_index, cont_covs=None, cat_covs=None, library=None, y=None, use_z_mean=False,):
        """Runs the generative model."""

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        # DECODER PEAKVI
        latent = z if not use_z_mean else qz_m
        decoder_input = (latent if cont_covs is None else torch.cat([latent, cont_covs], dim=-1))
        p = self.z_decoder(decoder_input, batch_index, *categorical_input)

        # DECODER SCVI
        decoder_input = z if cont_covs is None else torch.cat([z, cont_covs], dim=-1)
        px_scale, px_r, px_rate, px_dropout = self.decoder(self.dispersion, decoder_input, library, batch_index,
                                                           *categorical_input, y)
        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        return dict(p=p, px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout)

    def get_reconstruction_loss(self, p, d, f, x, px_rate, px_r, px_dropout):
        # Get Data
        x_rna = x[:, :self.n_input]
        x_chr = x[:, self.n_input:]

        # PEAKVI LOSS
        reconst_loss_peakvi = torch.nn.BCELoss(reduction="none")(p * d * f, (x_chr > 0).float()).sum(dim=-1)

        # SCVI LOSS
        reconst_loss_scvi = 0.
        if self.gene_likelihood == "zinb":
            reconst_loss_scvi = (-ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                                 .log_prob(x_rna).sum(dim=-1))
        elif self.gene_likelihood == "nb":
            reconst_loss_scvi = (-NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x_chr).sum(dim=-1))
        elif self.gene_likelihood == "poisson":
            reconst_loss_scvi = -Poisson(px_rate).log_prob(x_rna).sum(dim=-1)

        return reconst_loss_peakvi + reconst_loss_scvi

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        # GET DATA
        x = tensors[_CONSTANTS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        d = inference_outputs["d"]
        p = generative_outputs["p"]

        local_l_mean = tensors[_CONSTANTS.LOCAL_L_MEAN_KEY]
        local_l_var = tensors[_CONSTANTS.LOCAL_L_VAR_KEY]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        # KL DIV BETWEEN LOCAL VARIABLES
        kl_div_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1), ).sum(dim=1)
        if not self.use_observed_lib_size:
            kl_div_l = kl(Normal(ql_m, torch.sqrt(ql_v)), Normal(local_l_mean, torch.sqrt(local_l_var)),).sum(dim=1)
        else:
            kl_div_l = 0.0

        f = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        reconst_loss = self.get_reconstruction_loss(p, d, f, x, px_rate, px_r, px_dropout)

        kl_local_for_warmup = kl_div_z
        kl_local_no_warmup = kl_div_l
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)  # LOSS SCVI

        kl_local = dict(kl_divergence_l=kl_div_l, kl_divergence_z=kl_div_z)
        kl_global = 0.0
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)
