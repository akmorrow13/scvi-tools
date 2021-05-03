from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kl

from scvi import _CONSTANTS
from scvi._compat import Literal
from scvi.module.base import BaseModuleClass, LossRecorder, auto_move_data
from scvi.nn import Encoder, FCLayers


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


class TFVAE(BaseModuleClass):
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
        n_input_regions: int,
        m: int,
        n_batch: int = 0,
        n_hidden: Optional[int] = None,
        n_latent: Optional[int] = None,
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
    ):
        super().__init__()

        self.m = torch.tensor(np.transpose(m)).type(torch.float)
        self.n_tfs = m.shape[1]
        self.n_input_regions = n_input_regions
        self.n_hidden = (
            int(np.sqrt(self.n_input_regions)) if n_hidden is None else n_hidden)
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

        n_input_encoder = self.n_input_regions + n_continuous_cov * encode_covariates
        encoder_cat_list = cat_list if encode_covariates else None



        # ENCODER AND DECODER NNs FOR CELLS
        self.z_encoder = Encoder(n_input=n_input_encoder, n_layers=self.n_layers_encoder,
            n_output=self.n_latent, n_hidden=self.n_hidden, n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate, activation_fn=torch.nn.LeakyReLU, distribution=self.latent_distribution,
            var_eps=0, use_batch_norm=self.use_batch_norm_encoder, use_layer_norm=self.use_layer_norm_encoder)

        self.z_decoder = Decoder(n_input=self.n_latent + self.n_continuous_cov, n_output=n_input_regions,
            n_hidden=self.n_hidden, n_cat_list=cat_list, n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder, use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates)

        self.d_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.d_encoder = Decoder(n_input=n_input_encoder, n_output=1, n_hidden=self.n_hidden,
                n_cat_list=encoder_cat_list,n_layers=self.n_layers_encoder)

        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

        # ENCODER AND DECODER NNs FOR TFS
        self.chi_encoder = Encoder( n_input=n_input_encoder, n_layers=self.n_layers_encoder, n_output=self.n_latent,
            n_hidden=self.n_hidden, n_cat_list=encoder_cat_list, dropout_rate=self.dropout_rate,
            activation_fn=torch.nn.LeakyReLU, distribution=self.latent_distribution, var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder, use_layer_norm=self.use_layer_norm_encoder)

        self.chi_decoder = Decoder(n_input=self.n_latent + self.n_continuous_cov, n_output=self.n_input_regions,
            n_hidden=self.n_hidden, n_cat_list=cat_list, n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder, use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates)

        self.h_encoder = None
        if self.model_depth:
            # Decoder class to avoid variational split
            self.h_encoder = Decoder(n_input=n_input_encoder, n_output=1, n_hidden=self.n_hidden,
                                     n_cat_list=encoder_cat_list, n_layers=self.n_layers_encoder)

    def _get_inference_input(self, tensors):
        x = tensors[_CONSTANTS.X_KEY]
        m = self.m
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        cont_covs = tensors.get(_CONSTANTS.CONT_COVS_KEY)
        cat_covs = tensors.get(_CONSTANTS.CAT_COVS_KEY)
        input_dict = dict(
            x=x,
            m=m,
            batch_index=batch_index,
            cont_covs=cont_covs,
            cat_covs=cat_covs,
        )
        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        m,
        batch_index,
        cont_covs,
        cat_covs,
        n_samples=1,
    ) -> Dict[str, torch.Tensor]:
        """Helper function used in forward pass."""
        if cat_covs is not None and self.encode_covariates is True:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()
        if cont_covs is not None and self.encode_covariates is True:
            encoder_input = torch.cat([x, cont_covs], dim=-1)
            tf_encoder_input = torch.cat([m, cont_covs], dim=-1)
        else:
            encoder_input = x
            tf_encoder_input = m

        # if encode_covariates is False, cat_list to init encoder is None, so
        # batch_index is not used (or categorical_input, but it's empty)
        #       CELLS
        qz_m, qz_v, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        d = (self.d_encoder(encoder_input, batch_index, *categorical_input)
             if self.model_depth
             else 1)
        #       TFS
        qz_chi_m, qz_chi_v, chi = self.chi_encoder(tf_encoder_input, batch_index, *categorical_input)
        h = (self.d_encoder(encoder_input, batch_index, *categorical_input)
             if self.model_depth
             else 1)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            # when z is normal, untran_z == z
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)

            qz_chi_m = qz_chi_m.unsqueeze(0).expand((n_samples, qz_chi_m.size(0), qz_chi_m.size(1)))
            qz_chi_v = qz_chi_v.unsqueeze(0).expand((n_samples, qz_chi_v.size(0), qz_chi_v.size(1)))
            # when z is normal, untran_z == z
            untran_chi = Normal(qz_chi_m, qz_chi_v.sqrt()).sample()
            chi = self.chi_encoder.z_transformation(untran_chi)

        return dict(d=d, qz_m=qz_m, qz_v=qz_v, z=z, h=h, qz_chi_m=qz_chi_m, qz_chi_v=qz_chi_v, chi=chi)

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz_m"]
        chi = inference_outputs["chi"]
        qchi_m = inference_outputs["qz_chi_m"]
        batch_index = tensors[_CONSTANTS.BATCH_KEY]
        cont_covs = tensors.get(_CONSTANTS.CONT_COVS_KEY)
        cat_covs = tensors.get(_CONSTANTS.CAT_COVS_KEY)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
        input_dict = {
            "z": z,
            "qz_m": qz_m,
            "chi": chi,
            "qchi_m": qchi_m,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }
        return input_dict

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        chi,
        qchi_m,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        use_z_mean=False,
        use_chi_mean=False,
    ):
        """Runs the generative model."""

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        latent = z if not use_z_mean else qz_m
        z_decoder_input = (latent if cont_covs is None else torch.cat([latent, cont_covs], dim=-1))

        latentchi = chi if not use_chi_mean else qchi_m
        chi_decoder_input = (latentchi if cont_covs is None else torch.cat([latentchi, cont_covs], dim=-1))

        p = self.z_decoder(z_decoder_input, batch_index, *categorical_input)
        lamb = self.chi_decoder(chi_decoder_input, batch_index, *categorical_input)

        return dict(p=p, lamb=lamb)

    def get_reconstruction_loss(self, p, d, f, x, tfbs, tf_rate) -> torch.Tensor:
        reconst_loss = torch.nn.BCELoss(reduction="none")(p * d * f, (x > 0).float()).sum(dim=-1)

        reconst_loss += -Poisson(tf_rate).log_prob(tfbs).sum(dim=-1)

        return reconst_loss

    def loss(
        self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0
    ):
        # Getting Tensor Information
        x = tensors[_CONSTANTS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        d = inference_outputs["d"]
        p = generative_outputs["p"]

        tfbs = tensors[_CONSTANTS.PROTEIN_EXP_KEY]
        qz_chi_m = inference_outputs["qz_chi_m"]
        qz_chi_v = inference_outputs["qz_chi_v"]
        h = inference_outputs["h"]
        lamb = generative_outputs["lamb"]

        # Computing Elbo Terms
        kl_cells = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(0, 1)).sum(dim=1)
        kl_tfs = kl(Normal(qz_chi_m, torch.sqrt(qz_chi_v)), Normal(0, 1)).sum(dim=1)

        f = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        tf_rate = h * torch.matmul(p, lamb.transpose(0, 1))

        rl = self.get_reconstruction_loss(p, d, f, x, tfbs, tf_rate)

        loss = rl.sum() + kl_cells.sum() * kl_weight + kl_tfs.sum() * kl_weight

        return LossRecorder(loss, rl, kl_cells, kl_global=0.0)