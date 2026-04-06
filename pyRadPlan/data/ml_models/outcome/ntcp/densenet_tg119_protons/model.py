"""
DenseNet3D V5 - Dose-Preserving Architecture.

Key changes from V4:
1. BatchNorm3d to preserve absolute dose magnitude across samples
2. Less pooling - only 2 transitions instead of 3
3. Anisotropic pooling - pools XY more than Z (matches CT spacing)
4. Same API as V4 for drop-in replacement
"""

import torch
from torch import nn
from typing import Optional, Tuple, Union


class DenseBlockV5(nn.Module):
    """DenseBlock with BatchNorm for dose-magnitude preservation."""

    def __init__(
        self,
        num_convs: int,
        in_channels: int,
        growth_rate: int,
        drop_rate: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.drop_rate = drop_rate

        for i in range(num_convs):
            current_channels = in_channels + i * growth_rate
            # Pre-activation: BatchNorm -> SiLU -> Conv
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm3d(current_channels),
                    nn.SiLU(inplace=True),
                    nn.Conv3d(
                        current_channels,
                        growth_rate,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            y = block(x)
            if self.drop_rate > 0 and self.training:
                y = nn.functional.dropout(y, p=self.drop_rate, training=True)
            x = torch.cat((x, y), dim=1)
        return x


class BlurPool3d(nn.Module):
    """Anti-aliased downsampling: Gaussian blur + strided subsampling.

    Uses a fixed binomial-3 ([1,2,1]) kernel to low-pass filter before
    subsampling, preventing checkerboard artifacts in gradients.
    Based on Zhang (2019), "Making Convolutional Networks Shift-Invariant Again".
    """

    def __init__(self, channels: int, stride: Tuple[int, ...] = (1, 2, 2)):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = 1  # for 3x3x3 kernel

        # Binomial-3 tent kernel — gives uniform gradient scaling with stride 2
        a = torch.tensor([1.0, 2.0, 1.0])
        filt = a[:, None, None] * a[None, :, None] * a[None, None, :]
        filt = filt / filt.sum()
        filt = filt[None, None].repeat(channels, 1, 1, 1, 1)
        self.register_buffer("filt", filt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv3d(
            nn.functional.pad(x, [self.padding] * 6, mode="reflect"),
            self.filt,
            stride=self.stride,
            groups=self.channels,
        )


class DenseNet3DSmooth(nn.Module):
    """
    DenseNet3D V5 - Same API as V4 but with dose-preserving changes.

    Changes from V4:
    - BatchNorm3d to preserve absolute dose magnitude
    - Anisotropic pooling (less Z pooling)
    - 3 dense blocks with 2 transitions (less aggressive pooling)
    - Anti-aliased downsampling (BlurPool) for smooth dose gradients
    """

    def __init__(
        self,
        *,
        in_channels_ct: int = 1,
        in_channels_dose: int = 1,
        in_channels_mask: int = 1,
        stem_out_channels_total: int = 96,
        growth_rate: int = 32,
        arch: Tuple[int, int, int] = (4, 4, 4),
        block_ct_mask_input_grads: bool = True,
        drop_rate: float = 0.0,
        use_skip_connections: bool = True,
        batch_norm_dose: bool = True,
    ) -> None:
        super().__init__()

        if in_channels_dose < 1:
            raise ValueError("in_channels_dose must be >= 1")

        use_ct = in_channels_ct > 0
        use_mask = in_channels_mask > 0
        n_stems = 1 + int(use_ct) + int(use_mask)

        if stem_out_channels_total < n_stems:
            raise ValueError(f"stem_out_channels_total must be >= {n_stems}")

        base = stem_out_channels_total // n_stems
        rem = stem_out_channels_total - base * n_stems
        stem_dose_out = base + (1 if rem > 0 else 0)
        stem_ct_out = (base + (1 if rem > 1 else 0)) if use_ct else 0
        stem_mask_out = base if use_mask else 0

        self.in_channels_ct = in_channels_ct
        self.in_channels_dose = in_channels_dose
        self.in_channels_mask = in_channels_mask
        self.block_ct_mask_input_grads = block_ct_mask_input_grads
        self.use_skip_connections = use_skip_connections
        self.batch_norm_dose = batch_norm_dose

        # --- Anisotropic Stem with BatchNorm + anti-aliased downsampling ---
        def make_stem(in_ch: int, out_ch: int, use_batch_norm: bool = True) -> nn.Sequential:
            layers = [
                nn.Conv3d(
                    in_ch,
                    out_ch,
                    kernel_size=(3, 7, 7),
                    stride=1,  # No stride — avoids checkerboard gradients
                    padding=(1, 3, 3),
                ),
            ]
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(out_ch))
            layers.extend(
                [
                    nn.SiLU(inplace=True),
                    # Anti-aliased 4x downsampling in H,W (two 2x stages)
                    BlurPool3d(out_ch, stride=(1, 2, 2)),
                    BlurPool3d(out_ch, stride=(1, 2, 2)),
                ]
            )
            return nn.Sequential(*layers)

        self.stem_ct = make_stem(in_channels_ct, stem_ct_out) if use_ct else None
        self.stem_dose = make_stem(in_channels_dose, stem_dose_out, use_batch_norm=batch_norm_dose)
        self.stem_mask = make_stem(in_channels_mask, stem_mask_out) if use_mask else None

        init_channels = (
            stem_dose_out + (stem_ct_out if use_ct else 0) + (stem_mask_out if use_mask else 0)
        )

        # Anti-aliased transition: BlurPool instead of AvgPool for smooth gradients
        def make_transition(in_ch: int, out_ch: int, pool_z: bool = False) -> nn.Sequential:
            stride = (2, 2, 2) if pool_z else (1, 2, 2)
            return nn.Sequential(
                nn.BatchNorm3d(in_ch),
                nn.SiLU(inplace=True),
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
                BlurPool3d(out_ch, stride=stride),
            )

        # Build dense blocks and transitions (3 blocks, 2 transitions)
        self.dense_blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()
        self.skip_projections = nn.ModuleList() if use_skip_connections else None

        out_channels = init_channels
        self._block_output_channels = [init_channels]

        # Pool schedule: first transition XY only, second transition includes Z
        pool_z_schedule = [False, True]

        for i, num_convs in enumerate(arch):
            self.dense_blocks.append(
                DenseBlockV5(num_convs, out_channels, growth_rate, drop_rate=drop_rate)
            )
            out_channels += num_convs * growth_rate

            if i != len(arch) - 1:
                new_out = out_channels // 2
                pool_z = pool_z_schedule[i] if i < len(pool_z_schedule) else True

                self.transitions.append(make_transition(out_channels, new_out, pool_z=pool_z))

                if use_skip_connections:
                    skip_stride = (2, 2, 2) if pool_z else (1, 2, 2)
                    self.skip_projections.append(
                        nn.Sequential(
                            nn.Conv3d(self._block_output_channels[-1], new_out, kernel_size=1),
                            BlurPool3d(new_out, stride=skip_stride),
                        )
                    )

                out_channels = new_out
                self._block_output_channels.append(out_channels)

        self._final_channels = out_channels

        # Final head
        self.final_norm = nn.BatchNorm3d(out_channels)
        self.final_act = nn.SiLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(out_channels, 1)

        self.apply(self._custom_init_weights)
        self._dose_input_ref: Optional[torch.Tensor] = None

    def forward(
        self,
        x_or_dose: Union[torch.Tensor, None],
        mask: Optional[torch.Tensor] = None,
        ct: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - same API as V4."""
        if mask is None and ct is None:
            x = x_or_dose
            if x is None:
                raise ValueError("Input tensor is None")
            b, c, d, h, w = x.shape
            c_ct = self.in_channels_ct
            c_dose = self.in_channels_dose
            c_mask = self.in_channels_mask
            expected = c_ct + c_dose + c_mask
            if c != expected:
                raise ValueError(f"Expected {expected} channels, got {c}")
            off = 0
            x_ct = x[:, off : off + c_ct] if c_ct > 0 else None
            off += c_ct
            x_dose = x[:, off : off + c_dose]
            off += c_dose
            x_mask = x[:, off : off + c_mask] if c_mask > 0 else None
        else:
            x_dose = x_or_dose
            x_mask = mask
            x_ct = ct
            if x_dose is None:
                raise ValueError("Dose tensor must be provided")

        self._dose_input_ref = x_dose

        if self.block_ct_mask_input_grads:
            if x_ct is not None:
                x_ct = x_ct.detach()
            if x_mask is not None:
                x_mask = x_mask.detach()

        parts = []
        if self.stem_ct is not None:
            if x_ct is None:
                raise ValueError("CT input missing")
            parts.append(self.stem_ct(x_ct))

        parts.append(self.stem_dose(x_dose))

        if self.stem_mask is not None:
            if x_mask is None:
                raise ValueError("Mask input missing")
            parts.append(self.stem_mask(x_mask))

        x = torch.cat(parts, dim=1)

        skip_features = [x] if self.use_skip_connections else None

        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)

            if i < len(self.transitions):
                x = self.transitions[i](x)

                if self.use_skip_connections and i < len(self.skip_projections):
                    skip = self.skip_projections[i](skip_features[-1])
                    x = x + skip
                    skip_features.append(x)

        x = self.final_norm(x)
        x = self.final_act(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        return out

    @staticmethod
    def _custom_init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm3d):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def enable_dose_gradient_mode(self) -> None:
        self.eval()
        self.block_ct_mask_input_grads = True
