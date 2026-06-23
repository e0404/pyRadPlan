try:
    import torch
    from torch import nn
except ImportError:
    torch = None

import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBayes_new(nn.Module):
    def __init__(
        self,
        batch_size,
        nstep,
        imsize,
        projection_dim,
        n_neuron=100,
        nlayer=1,
        energy_encoding=False,
        reduced_model=False,
        use_mlp=False,
    ):
        super(ConvBayes_new, self).__init__()

        self.n_neurons = n_neuron
        self.batch_size = batch_size
        self.n_step = nstep
        self.imsize = imsize
        self.n_inputs = projection_dim[0] * projection_dim[1] * projection_dim[2]
        self.n_outputs = projection_dim[0] * projection_dim[1] * projection_dim[2]
        self.n_layers = nlayer
        self.num_tokens = nstep  # + 1
        self.projection_dim = projection_dim
        self.energy_encoding = energy_encoding
        self.energy_token = None
        self.use_mlp = use_mlp

        self.norm = nn.LayerNorm(self.n_outputs, eps=1e-6)
        self.mlp = MLP(self.n_outputs)

        if reduced_model:
            self.conv_encoder = ConvEncoder_reduced(
                self.num_tokens,
                self.projection_dim,
                self.batch_size,
                pos_encoding=False,
                energy_encoding=self.energy_encoding,
            )
            self.lstm = nn.LSTM(self.n_inputs, self.n_outputs, batch_first=True)
            self.conv_decoder = ConvDecoder_reduced(
                self.projection_dim, energy_encoding=self.energy_encoding
            )
        else:
            self.conv_encoder = ConvEncoder(
                self.num_tokens,
                self.projection_dim,
                self.batch_size,
                pos_encoding=False,
                energy_encoding=self.energy_encoding,
            )
            self.lstm = nn.LSTM(self.n_inputs, self.n_outputs, batch_first=True)
            self.conv_decoder = ConvDecoder(
                self.projection_dim, energy_encoding=self.energy_encoding
            )

        # self.backend = nn.ReLU()

    def forward(self, x):
        if x.ndim == 4:
            if self.energy_encoding is True:
                self.energy_token = x[:, :1, 0, 0].squeeze(1)
                x = x[:, 1:, :, :]
            x = x.unsqueeze(1)

        elif x.ndim == 3:
            if self.energy_encoding is True:
                self.energy_token = x[:1, 0, 0]
                x = x[1:, :, :]
            x = x.unsqueeze(0).unsqueeze(1)

        x = self.conv_encoder(tokens=x, energy_token=self.energy_token)

        if getattr(self, "use_mlp", False):
            mlp_out = self.mlp(self.norm(x))
            x = x + mlp_out

        lstm_out, temp = self.lstm(x)

        # if self.energy_encoding is True:
        #    out, e = self.conv_decoder(lstm_out)
        #    out = torch.cat((e.view(e.shape[0], 1, 1, 1, 1).repeat(1, 1, 1, out.shape[3], out.shape[4]), out), dim = 2).float()

        # else:
        out = self.conv_decoder(lstm_out)

        # out = self.backend(out)

        out = out.squeeze(1)

        return out


class TimeDistributed(nn.Module):
    """Time distributed layer (applies the same module on all time steps of a sequence)"""

    def __init__(self, module, tensor_size=5):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.tensor_size = tensor_size
        self.transmutation_pos = int((self.tensor_size - 1) / 2)

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        x = torch.transpose(x, 0, self.transmutation_pos)

        if self.tensor_size == 5:
            x = torch.transpose(x, 1, self.transmutation_pos)

        y = torch.stack([self.module(x[i]) for i in range(x.size()[0])])

        if self.tensor_size == 5:
            y = torch.transpose(x, 1, self.transmutation_pos)

        y = torch.transpose(y, 0, self.transmutation_pos)

        return y


class ConvEncoder(nn.Module):
    """Learnable position embedding + initial projection"""

    def __init__(
        self,
        num_tokens,
        projection_dim,
        batch_size,
        num_channels=64,
        activation_func=nn.ReLU,
        kernel_size=(1, 3, 3),
        strides=1,
        pooling=(1, 2, 2),
        pos_encoding=True,
        energy_encoding=False,
    ):
        super(ConvEncoder, self).__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim
        self.projection_dim_flat = np.prod(projection_dim)
        self.projection_channels = projection_dim[0]  # channels first
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.pooling = pooling
        self.pos_encoding = pos_encoding
        self.activation_func = activation_func
        self.energy_encoding = energy_encoding
        # self.additional_layer = additional_layer

        # Convolutional layers
        self.conv1 = nn.Conv3d(
            1, self.num_channels, kernel_size=self.kernel_size, stride=self.strides
        )  # padding=(0,1,1)) for non-mu dataset
        self.conv2 = nn.Conv3d(
            self.num_channels,
            self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=(0, 1, 1),
        )

        # Final convolutional
        self.conv3 = nn.Conv3d(
            self.num_channels,
            self.projection_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=(0, 1, 1),
        )

        # Pooling layers
        self.pool1 = nn.MaxPool3d(
            kernel_size=self.pooling, stride=self.pooling
        )  # stride = pooling so pooling does not overlap between kernels
        self.pool2 = nn.MaxPool3d(kernel_size=self.pooling, stride=self.pooling)

        # Normalization layers
        self.norm1 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))
        self.norm2 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))
        self.norm3 = TimeDistributed(
            nn.GroupNorm(num_groups=4, num_channels=self.projection_channels)
        )

        # Activation layer
        self.h1 = TimeDistributed(self.activation_func())
        self.h2 = TimeDistributed(self.activation_func())
        self.h3 = TimeDistributed(self.activation_func())

        # if additional_layer == True:
        #    self.conv4 = nn.Conv3d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=self.strides, padding=(0,2,2))
        #    self.pool4 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,3,3))
        #    self.norm4 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))
        #    self.h4 = TimeDistributed(self.activation_func())

        # Flatten 3D slices to 1D vectors
        self.flatten = TimeDistributed(nn.Flatten(1, 3), tensor_size=3)

        # Positional encoding
        self.positional_encoder = nn.Embedding(
            num_embeddings=self.num_tokens, embedding_dim=self.projection_dim_flat
        )

        # Linear transformation of the energy
        self.linear = nn.Linear(
            in_features=1, out_features=self.projection_dim_flat
        )  # W_0 in paper (e = W_O * E), E is tensor of size batch_size

    def forward(self, tokens, energy_token=None):
        # First convolutional block

        x = self.h1(self.pool1(self.norm1(self.conv1(tokens))))

        # Second convolutional block

        x = self.h2(self.pool2(self.norm2(self.conv2(x))))

        # if self.additional_layer == True:
        #    x = self.h4(self.pool4(self.norm4(self.conv4(x))))

        # Last convolution
        x = self.h3(self.norm3(self.conv3(x)))

        # Flatten to 1D vector (permute for "channels_last")
        x = self.flatten(
            x.permute(0, 2, 3, 4, 1)
        )  # permute such that: batches, sequence steps, channels, spatial dimensions

        if energy_token is not None:
            # Encode energy and concatenate
            e = self.linear(energy_token.float().unsqueeze(1)).unsqueeze(1)
            x = torch.cat((e, x), dim=1)

        # Add positional encoding
        if self.pos_encoding:
            positions = (
                torch.arange(start=0, end=self.num_tokens).int().to(device)
            )  # num_tokens-1 because no energy added
            encoded = x + self.positional_encoder(positions)
            return encoded
        else:
            return x


class ConvDecoder(nn.Module):
    """Convert transformer output to dose."""

    """ Padding """

    def __init__(
        self,
        projection_dim,
        energy_encoding=False,
        num_channels=64,
        activation_func=nn.ReLU,
        kernel_size=(1, 3, 3),
        strides=(1, 2, 2),
        padding=(0, 1, 1),
    ):  # padding added for mu-dataset
        super(ConvDecoder, self).__init__()
        self.projection_dim = projection_dim
        self.projection_dim_flat = np.prod(projection_dim)
        self.projection_channels = projection_dim[0]  # channels first
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.energy_encoding = energy_encoding
        self.activation_func = activation_func
        # self.additional_layer = additional_layer
        # self.padding = padding

        # Reshape to 3D
        self.reshape = TimeDistributed(
            nn.Unflatten(
                dim=1, unflattened_size=[projection_dim[1], projection_dim[2], projection_dim[0]]
            ),
            tensor_size=3,
        )  # dim=1 do to TimeDistribution going through all batches --> 0:slice, 1:flattened tensor

        # Convolutional Transpose Layers
        self.conv1 = nn.ConvTranspose3d(
            self.projection_channels,
            self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
        )
        self.conv2 = nn.ConvTranspose3d(
            self.num_channels,
            self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            output_padding=(0, 1, 1),
        )  # output_padding for mu-dataset

        # if self.additional_layer == True:
        #    self.conv4 = nn.ConvTranspose3d(self.num_channels, self.num_channels, kernel_size=self.kernel_size, stride=self.strides, output_padding=(0,1,1))

        # Output convolutional
        self.conv3 = nn.Conv3d(
            self.num_channels, 1, kernel_size=self.kernel_size, stride=1
        )  # , padding=(0,1,1)

        # Normalization layers
        self.norm1 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))
        self.norm2 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))

        # Activation layer
        self.h1 = TimeDistributed(self.activation_func())
        self.h2 = TimeDistributed(self.activation_func())

        # Linear transformation of the energy
        self.linear = nn.Linear(
            in_features=self.projection_dim_flat, out_features=1
        )  # W_0 in paper (e = W_O * E), E is tensor of size batch_size

    def forward(self, tokens):
        if self.energy_encoding is True:
            # remove first token (energy token)
            tokens = tokens[:, 1:, :]
            # e = tokens[:, :1, :,].squeeze()

        # Reshape (permute for "channels_first")
        x = (self.reshape(tokens)).permute(0, 4, 1, 2, 3)

        # First convolutional block
        x = self.h1(self.norm1(self.conv1(x)))

        # Second convolutional block
        x = self.h2(self.norm2(self.conv2(x)))

        # if self.additional_layer == True:
        #    x = self.h2(self.norm2(self.conv4(x)))

        # Last convolution
        x = self.conv3(x)

        # if self.energy_encoding is True:
        #    # Encode energy and concatenate
        #    e = self.linear(e)

        return x  # ,e


class ConvEncoder_reduced(nn.Module):
    """Learnable position embedding + initial projection"""

    def __init__(
        self,
        num_tokens,
        projection_dim,
        batch_size,
        num_channels=64,
        kernel_size=(1, 7, 7),
        strides=(1, 4, 4),
        pooling=(1, 2, 2),
        pos_encoding=True,
        energy_encoding=False,
    ):
        super(ConvEncoder_reduced, self).__init__()
        self.batch_size = batch_size
        self.num_tokens = num_tokens
        self.projection_dim = projection_dim
        self.projection_dim_flat = np.prod(projection_dim)
        self.projection_channels = projection_dim[0]  # channels first
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.pooling = pooling
        self.pos_encoding = pos_encoding
        self.energy_encoding = energy_encoding

        # Convolutional layers
        self.conv1 = nn.Conv3d(
            1,
            self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=(0, 1, 1),
        )  # , padding='same'))

        # Final convolutional
        self.conv3 = nn.Conv3d(
            self.num_channels,
            self.projection_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=(0, 1, 1),
        )

        # Normalization layers
        self.norm1 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))
        self.norm3 = TimeDistributed(
            nn.GroupNorm(num_groups=4, num_channels=self.projection_channels)
        )

        # Activation layer
        self.h1 = TimeDistributed(nn.ReLU())
        self.h3 = TimeDistributed(nn.ReLU())

        # Flatten 3D slices to 1D vectors
        self.flatten = TimeDistributed(nn.Flatten(1, 3), tensor_size=3)

        # Positional encoding
        self.positional_encoder = nn.Embedding(
            num_embeddings=self.num_tokens, embedding_dim=self.projection_dim_flat
        )

        # Linear transformation of the energy
        self.linear = nn.Linear(
            in_features=1, out_features=self.projection_dim_flat
        )  # W_0 in paper (e = W_O * E), E is tensor of size batch_size

    def forward(self, tokens, energy_token=None):
        # First convolutional block
        x = self.h1(self.norm1(self.conv1(tokens)))

        # Last convolution
        x = self.h3(self.conv3(x))

        # Flatten to 1D vector (permute for "channels_last")
        x = self.flatten(x.permute(0, 2, 1, 3, 4))

        if energy_token is not None:
            # Encode energy and concatenate
            e = self.linear(energy_token.float().unsqueeze(1)).unsqueeze(1)
            x = torch.cat((e, x), dim=1)

        # Add positional encoding
        if self.pos_encoding:
            positions = (
                torch.arange(start=0, end=self.num_tokens).int().to(device)
            )  # num_tokens-1 because no energy added
            encoded = x + self.positional_encoder(positions)
            return encoded
        else:
            return x


class ConvDecoder_reduced(nn.Module):
    """Convert transformer output to dose."""

    """ Padding """

    def __init__(
        self,
        projection_dim,
        energy_encoding=False,
        num_channels=64,
        kernel_size=(1, 7, 7),
        strides=(1, 4, 4),
    ):
        super(ConvDecoder_reduced, self).__init__()
        self.projection_dim = projection_dim
        self.projection_channels = projection_dim[0]  # channels first
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.energy_encoding = energy_encoding

        # Reshape to 3D
        self.reshape = TimeDistributed(
            nn.Unflatten(
                dim=1, unflattened_size=[projection_dim[1], projection_dim[2], projection_dim[0]]
            ),
            tensor_size=3,
        )  # dim=1 do to TimeDistribution going through all batches --> 0:slice, 1:flattened tensor

        # Convolutional Transpose Layers
        # Configure conv-transpose to upscale from projection spatial dims (e.g. 3x3)
        # back to the original image size (e.g. 15x15). For projection 3 ->
        # output 15 use kernel=7, stride=4, padding=0 because:
        # (in-1)*stride - 2*pad + kernel = (3-1)*4 - 0 + 7 = 15
        self.conv1 = nn.ConvTranspose3d(
            self.projection_channels,
            self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=(0, 2, 2),
            output_padding=(0, 3, 3),
        )  # , output_padding=(0,1,1)) padding for MU dataset

        # Output convolutional
        self.conv3 = nn.Conv3d(
            self.num_channels, 1, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
        )

        # Normalization layers
        self.norm1 = TimeDistributed(nn.GroupNorm(num_groups=16, num_channels=self.num_channels))

        # Activation layer
        self.h1 = TimeDistributed(nn.ReLU())

    def forward(self, tokens):
        if self.energy_encoding:
            # remove first token (energy token)
            tokens = tokens[:, 1:, :]

        # Reshape (permute for "channels_first")
        x = (self.reshape(tokens)).permute(0, 4, 1, 2, 3)

        # First convolutional block
        x = self.h1(self.norm1(self.conv1(x)))

        # Last convolution
        x = self.conv3(x)

        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads,
        num_tokens,
        projection_dim,
        causal=True,
        dropout_rate=0.2,
        num_forward=0,
        use_MLP=False,
    ):
        super(TransformerEncoder, self).__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.projection_dim = np.prod(projection_dim)
        self.causal = causal
        self.use_MLP = use_MLP

        # Multi-head self attention layer
        self.multihead = nn.MultiheadAttention(
            embed_dim=self.projection_dim,
            num_heads=self.num_heads,
            dropout=dropout_rate,
            bias=False,
            batch_first=True,
        )

        # MLP
        if self.use_MLP:
            self.mlp_network = MLP(self.projection_dim, dropout_rate=dropout_rate)

        # Normalization layers
        self.norm1 = nn.LayerNorm(self.projection_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(self.projection_dim, eps=1e-6)

        # Mask for causal attention
        if causal:
            self.mask = (
                torch.triu(torch.ones((num_tokens, num_tokens)), diagonal=num_forward + 1)
                .bool()
                .to(device)
            )  # Lower triangular matrix to prevent model from cheating
            # self.mask[:,-1] = False
        else:
            self.mask = (
                torch.zeros((self.num_tokens, self.num_tokens)).bool().to(device)
            )  # each token can see all other tokens

    def forward(self, tokens):
        # always norm. before sub-layers
        x = self.norm1(tokens)
        x, w = self.multihead(x, x, x, attn_mask=self.mask)  # x 3 times as query, key, value
        x = x + tokens  # residual connection
        if self.use_MLP:
            y = self.norm2(x)
            y = self.mlp_network(y)
            return x + y
        else:
            return x


class MLP(nn.Module):
    """Multilayer Perceptron (MLP) module for DotaDose."""

    def __init__(self, projection_dim, dropout_rate=0.2):
        super(MLP, self).__init__()

        self.projection_dim = projection_dim

        self.mlp_network = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim),  # Linear transformation
            nn.GELU(),  # introduces non-linearity to learn complex patterns
            nn.Dropout(dropout_rate),
            nn.Linear(self.projection_dim, self.projection_dim),
            nn.GELU(),
            nn.Dropout(
                dropout_rate
            ),  # randomly sets fraction of input units to 0 to prevent overfitting
        )

    def forward(self, x):
        return self.mlp_network(x)
