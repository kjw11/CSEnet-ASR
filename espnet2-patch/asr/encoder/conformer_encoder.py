# Copyright 2020 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Conformer encoder definition."""

import logging
from typing import List, Optional, Tuple, Union

import torch
from typeguard import check_argument_types

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation, make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    TooShortUttError,
    check_short_utt,
)
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class ConformerEncoder(AbsEncoder):
    """Conformer encoder module.

    Args:
        input_size (int): Input dimension.
        output_size (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        attention_dropout_rate (float): Dropout rate in attention.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            If True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            If False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        rel_pos_type (str): Whether to use the latest relative positional encoding or
            the legacy one. The legacy relative positional encoding will be deprecated
            in the future. More Details can be found in
            https://github.com/espnet/espnet/pull/2816.
        encoder_pos_enc_layer_type (str): Encoder positional encoding layer type.
        encoder_attn_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_sd_blocks: int = 4,
        num_rec_blocks: int = 8,
        num_joint_blocks: int = 0,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer: str = "conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
        padding_idx: int = -1,
        interctc_layer_idx: List[int] = [],
        interctc_use_conditioning: bool = False,
        stochastic_depth_rate: Union[float, List[float]] = 0.0,
        speaker_emb: bool = True,
        joint_heat: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self._drop_rate = dropout_rate

        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(input_size, output_size),
                torch.nn.LayerNorm(output_size),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                input_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(input_size, output_size, padding_idx=padding_idx),
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        if selfattention_layer_type == "selfattn":
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
            logging.warning(
                "Using legacy_rel_selfattn and it will be deprecated in the future."
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)

        num_blocks = num_sd_blocks + num_rec_blocks + num_joint_blocks

        if isinstance(stochastic_depth_rate, float):
            stochastic_depth_rate = [stochastic_depth_rate] * num_blocks

        if len(stochastic_depth_rate) != num_blocks:
            raise ValueError(
                f"Length of stochastic_depth_rate ({len(stochastic_depth_rate)}) "
                f"should be equal to num_blocks ({num_blocks})"
            )

        self.shared_layer = Conv2dSubsampling(
                output_size,
                output_size,
                dropout_rate,
                pos_enc_class(output_size, positional_dropout_rate),
            )
        self.sd1 = repeat(
            num_sd_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum+2],
            ),
        )
        self.sd2 = repeat(
            num_sd_blocks,
            lambda lnum: EncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate[lnum+2],
            ),
        )

        if num_joint_blocks < 1:
            self.joint_blocks = None
        else:
            self.joint_blocks = repeat(
                num_joint_blocks,
                lambda lnum: EncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    stochastic_depth_rate[lnum+2],
                ),
            )
            self.pos_enc = pos_enc_class(output_size, positional_dropout_rate)

        if num_rec_blocks < 1:
            self.encoders = None
        else:
            self.encoders = repeat(
                num_rec_blocks,
                lambda lnum: EncoderLayer(
                    output_size,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    stochastic_depth_rate[lnum+num_joint_blocks],
                ),
            )


        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        self.interctc_layer_idx = interctc_layer_idx
        if len(interctc_layer_idx) > 0:
            assert 0 <= min(interctc_layer_idx) and max(interctc_layer_idx) < num_blocks
        self.interctc_use_conditioning = interctc_use_conditioning
        self.conditioning_layer = None

        # set a learnable parameter
        self.bound = torch.nn.Parameter(torch.zeros(1, 1, output_size))
        self.speaker_emb = speaker_emb
        if speaker_emb:
            logging.info("Use speaker embedding.")
            self.global_emb = torch.nn.Parameter(torch.zeros(1, output_size))
            self.spkr_emb1 = torch.nn.Parameter(torch.zeros(1, output_size))
            self.spkr_emb2 = torch.nn.Parameter(torch.zeros(1, output_size))

        self.joint_heat = joint_heat
        if joint_heat:
            logging.info("Use joint CTC.")

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc: CTC = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        batch_size = xs_pad.size(0)
        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)

        if (
            isinstance(self.embed, Conv2dSubsampling)
            or isinstance(self.embed, Conv2dSubsampling2)
            or isinstance(self.embed, Conv2dSubsampling6)
            or isinstance(self.embed, Conv2dSubsampling8)
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        intermediate_outs = []

        if isinstance(xs_pad, tuple):
            x, pos_emb = xs_pad[0], xs_pad[1]
        else:
            x, pos_emb = xs_pad, None

        # 1: shared layers
        xs_pad, masks = self.shared_layer(x, masks)
        global_xs_pad = xs_pad[0]
        global_lens = masks.squeeze(1).sum(1)

        # 2: speaker encoder
        h1_out, masks = self.sd1(xs_pad, masks)
        h2_out, masks = self.sd2(xs_pad, masks)

        assert isinstance(h1_out, tuple)
        h1, pos_emb = h1_out[0], h1_out[1]
        h2, pos_emb = h2_out[0], h2_out[1]
        lens = masks.squeeze(1).sum(1)

        # 3: joint layers
        if self.joint_blocks is not None:

            # concat out1, bound, out2 along dim=1
            bound = self.bound.expand(batch_size, 1, -1)

            concat_out = []
            # unpad and concat
            for i in range(batch_size):
                global_unpad = global_xs_pad[i, :global_lens[i]]
                h1_unpad = h1[i, :lens[i]]
                h2_unpad = h2[i, :lens[i]]

                if self.speaker_emb:
                    # add speaker embedding
                    global_emb = self.global_emb.expand(global_lens[i], -1)
                    spkr_emb1 = self.spkr_emb1.expand(lens[i], -1)
                    spkr_emb2 = self.spkr_emb2.expand(lens[i], -1)
                    global_unpad = global_unpad + global_emb
                    h1_unpad = h1_unpad + spkr_emb1
                    h2_unpad = h2_unpad + spkr_emb2

                # concat
                concat_out.append(
                    torch.cat((global_unpad, h1_unpad, bound[i], h2_unpad), dim=0)
                    )

            # pad again
            concat_out = torch.nn.utils.rnn.pad_sequence(concat_out, batch_first=True)
            concat_out_lens = global_lens + lens * 2 + 1
            masks_cat = (~make_pad_mask(concat_out_lens)[:, None, :]).to(concat_out.device)

            # 3.5 deal with interCTC
            if len(self.interctc_layer_idx) != 0:
                # only consider the separated output
                assert len(self.interctc_layer_idx) == 1 and self.interctc_layer_idx[0] == 0
                if self.normalize_before:
                    h1 = self.after_norm(h1)
                    h2 = self.after_norm(h2)
                    if self.joint_blocks:
                        # concat h1 and h2
                        concat_out = torch.cat([h1,h2], dim=1)
                    concat_out = self.after_norm(concat_out)
                intermediate_outs.append((h1, h2, concat_out))
                assert ~self.interctc_use_conditioning


            # forward joint layers
            _, pos_emb_cat = self.pos_enc(concat_out)
            xs_pad = (concat_out, pos_emb_cat)
            masks = masks_cat

            xs_pad, masks = self.joint_blocks(xs_pad, masks)

            if self.encoders is not None:
                h_cat, pos_emb_cat = xs_pad[0], xs_pad[1]
                h1 = []
                h2 = []
                for i in range(batch_size):
                    h1.append(h_cat[i, global_lens[i]:global_lens[i]+lens[i]])
                    h2.append(h_cat[i, global_lens[i]+lens[i]+1:concat_out_lens[i]])
                h1 = torch.nn.utils.rnn.pad_sequence(h1, batch_first=True)
                h2 = torch.nn.utils.rnn.pad_sequence(h2, batch_first=True)
            
        # 4: recognition layers
        if self.encoders is not None:
            # forward encoders
            # put in batch dim for acceleration
            h_split = torch.cat((h1, h2), dim=0)
            masks = (~make_pad_mask(lens)[:, None, :]).to(h1.device)
            masks_split = torch.cat((masks, masks), dim=0)
            # forward
            xs_pad, masks = self.encoders((h_split, pos_emb), masks_split)

            if self.joint_heat:
                # cat again
                h1, h2 = torch.split(xs_pad[0], batch_size, dim=0)
                bound = self.bound.expand(batch_size, 1, -1)
                xs_cat = []
                for i in range(batch_size):
                    h1_unpad = h1[i, :lens[i]]
                    h2_unpad = h2[i, :lens[i]]
                    xs_cat.append(
                        torch.cat((h1_unpad, bound[i], h2_unpad), dim=0)
                        )
                xs_pad = torch.nn.utils.rnn.pad_sequence(xs_cat, batch_first=True)
        else:
            assert self.joint_blocks is not None
            # remove global
            h_cat = xs_pad[0]
            xs_cat = []
            for i in range(batch_size):
                xs_cat.append(h_cat[i, global_lens[i]:concat_out_lens[i]])
            xs_cat = torch.nn.utils.rnn.pad_sequence(xs_cat, batch_first=True)
            xs_pad = xs_cat

        masks = (~make_pad_mask(lens + lens + 1)[:, None, :]).to(xs_pad.device)

        if isinstance(xs_pad, tuple):
            xs_pad = xs_pad[0]
        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None
