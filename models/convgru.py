import torch
import torch.nn as nn
from torch.nn import init
from transformers import GPT2Config, GPT2Model


class Transformer(nn.Module):
    def __init__(self, in_channel, args):
        super(Transformer, self).__init__()
        self.in_channel = in_channel
        self.encoder = nn.Linear(self.in_channel, args.inter_dim)
        self.gru = GPT2Model(
            GPT2Config(n_embd=args.inter_dim,
                       vocab_size=768,
                       use_cache=True,
                       n_layer=args.num_layers_gru,
                       n_head=args.num_heads_gru,
                       output_attentions=False))
        del self.gru.wte

    def forward(self, feat):
        if feat.ndim == 6:
            # Input feat is of shape batch * temp * num_box * x * x
            _b, _t, _n, _c, _h, _w = feat.shape
            feat = feat.permute(0, 2, 1, 3, 4, 5)  # _b, _n, _t, x, x
            feat = feat.reshape(_b * _n, _t, -1)
        elif feat.ndim == 5:
            _b, _t, _n, _c, _h = feat.shape
            feat = feat.permute(0, 2, 1, 3, 4)
            feat = feat.reshape(_b * _n, _t, -1)

        prev = None
        feats = self.encoder(feat)
        position_ids = torch.arange(0,
                                    feats.size(1),
                                    dtype=torch.long,
                                    device=feats.device)
        outputs = self.gru(inputs_embeds=feats,
                           past_key_values=prev,
                           position_ids=position_ids)
        out = outputs.last_hidden_state
        
        # attention = outputs.attentions
        # prev = outputs.past_key_values
        return out


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 kernel_size,
                 bias_i=True,
                 bias_h=True):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_i = bias_i
        self.bias_h = bias_h

        self.update_i = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias_i,
        )
        self.update_h = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias_h,
        )

        self.reset_i = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias_i,
        )
        self.reset_h = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias_h,
        )

        self.out_i = nn.Conv2d(
            input_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias_i,
        )
        self.out_h = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias_h,
        )

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self._init_params()

    def _init_params(self):
        init.kaiming_normal(self.update_i.weight)
        init.kaiming_normal(self.update_h.weight)
        init.kaiming_normal(self.reset_i.weight)
        init.kaiming_normal(self.reset_h.weight)
        init.kaiming_normal(self.out_i.weight)
        init.kaiming_normal(self.out_h.weight)

        if self.bias_i:
            init.constant_(self.reset_i.bias, 0.0)
            init.constant_(self.update_i.bias, 0.0)
            init.constant_(self.out_i.bias, 0.0)
        if self.bias_h:
            init.constant_(self.reset_h.bias, 0.0)
            init.constant_(self.update_h.bias, 0.0)
            init.constant_(self.out_h.bias, 0.0)

    def repackage_hidden(self, h):
        """Wraps hidden states in new Tensors, to detach them from their
        history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def forward(self, x, prev_state):
        # data size is [batch, channel, height, width]

        # get batch and spatial sizes
        batch_size = x.size(0)
        spatial_size = x.shape[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(*state_size).type_as(x)
        else:
            prev_state = self.repackage_hidden(prev_state)

        z_i = self.update_i(x)
        z_h = self.update_h(prev_state).requires_grad_()
        z = self.sigmoid(z_i + z_h)

        r_i = self.reset_i(x)
        r_h = self.reset_h(prev_state).requires_grad_()
        r = self.sigmoid(r_i + r_h)

        h_i = self.out_i(x)
        h_h = self.out_h(r * prev_state).requires_grad_()
        out = self.tanh(h_i + h_h)

        new_state = ((1 - z) * prev_state) + (z * out)
        return new_state


class ConvGRU(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_sizes, n_layers):
        """
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        """

        super(ConvGRU, self).__init__()

        self.input_size = input_size

        if not isinstance(hidden_sizes, list):
            self.hidden_sizes = [hidden_sizes] * n_layers
        else:
            assert (len(hidden_sizes) == n_layers
                    ), "`hidden_sizes` must have the same length as n_layers"
            self.hidden_sizes = hidden_sizes
        if not isinstance(kernel_sizes, list):
            self.kernel_sizes = [kernel_sizes] * n_layers
        else:
            assert (len(kernel_sizes) == n_layers
                    ), "`kernel_sizes` must have the same length as n_layers"
            self.kernel_sizes = kernel_sizes

        self.n_layers = n_layers

        self.cells = []
        for i in range(self.n_layers):
            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i - 1]

            cell = ConvGRUCell(input_dim, self.hidden_sizes[i],
                               self.kernel_sizes[i])

            self.cells.append(cell)
        self.cells = nn.ModuleList(self.cells)

    def forward(self, x, hidden=None):
        """
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels,
        height, width).

        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels,
        height, width).
        """
        if (hidden is None) or (len(hidden) == 0):
            hidden = [None] * self.n_layers

        upd_hidden = []

        for layer_idx in range(self.n_layers):
            cell = self.cells[layer_idx]
            cell_hidden = hidden[layer_idx]

            # pass through layer
            upd_cell_hidden = cell(x, cell_hidden)
            upd_hidden.append(upd_cell_hidden)
            # update input_ to the last updated hidden layer for next pass
            x = upd_cell_hidden

        # retain tensors in list to allow different hidden sizes
        return upd_hidden, upd_hidden[-1]
