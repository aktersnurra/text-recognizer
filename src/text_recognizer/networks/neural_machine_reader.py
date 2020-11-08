from typing import Dict, Optional, Tuple

from einops import rearrange
from einops.layers.torch import Rearrange
import torch
from torch import nn
from torch import Tensor

from text_recognizer.networks.util import configure_backbone


class Encoder(nn.Module):

    def __init__(self,  embedding_dim: int, encoder_dim: int, decoder_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_size=embedding_dim, hidden_size=encoder_dim, bidirectional=True)
        self.fc = nn.Sequential(nn.Linear(in_features=2*encoder_dim, out_features=decoder_dim), nn.Tanh())
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encodes a sequence of tensors with a bidirectional GRU.

        Args:
            x (Tensor): A input sequence.

        Shape:
            - x: :math:`(T, N, E)`.
            - output[0]: :math:`(T, N, 2 * E)`.
            - output[1]: :math:`(T, N, D)`.

            where T is the sequence length, N is the batch size, E is the
            embedding/encoder dimension, and D is the decoder dimension.

        Returns:
            Tuple[Tensor, Tensor]: The encoder output and the hidden state of the
                encoder.

        """

        output, hidden = self.rnn(x)

        # Get the hidden state from the forward and backward rnn.
        hidden_state = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))

        # Apply fully connected layer and tanh activation.
        hidden_state = self.fc(hidden_state)

        return output, hidden_state


class Attention(nn.Module):

    def __init__(self, encoder_dim: int, decoder_dim: int) -> None:
        super().__init__()
        self.atten = nn.Linear(in_features=2*encoder_dim + decoder_dim, out_features=decoder_dim)
        self.value = nn.Linear(in_features=decoder_dim, out_features=1, bias=False)

    def forward(self, hidden_state: Tensor, encoder_outputs: Tensor) -> Tensor:
        """Short summary.

        Args:
            hidden_state (Tensor): Description of parameter `h`.
            encoder_outputs (Tensor): Description of parameter `enc_out`.

        Shape:
            - x: :math:`(T, N, E)`.
            - output[0]: :math:`(T, N, 2 * E)`.
            - output[1]: :math:`(T, N, D)`.

            where T is the sequence length, N is the batch size, E is the
            embedding/encoder dimension, and D is the decoder dimension.

        Returns:
            Tensor: Description of returned object.

        """
        t, b = enc_out.shape[:2]
        #repeat decoder hidden state src_len times
        hidden_state = hidden_state.unsqueeze(1).repeat(1, t, 1)

        encoder_outputs = rearrange(encoder_outputs, "t b e2 -> b t e2")

        # Calculate the energy between the decoders previous hidden state and the
        # encoders hidden states.
        energy = torch.tanh(self.attn(torch.cat((hidden_state, encoder_outputs), dim = 2)))

        attention = self.value(energy).squeeze(2)

        # Apply softmax on the attention to squeeze it between 0 and 1.
        attention = F.softmax(attention, dim=1)

        return attention


class Decoder(nn.Module):

    def __init__(self, embedding_dim: int, encoder_dim: int, decoder_dim: int, output_dim: int, dropout_rate: float = 0.1) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.attention = Attention(encoder_dim, decoder_dim)
        self.rnn = nn.GRU(input_size=2*encoder_dim + embedding_dim, hidden_size=decoder_dim)

        self.head = nn.Linear(in_features=2*encoder_dim+embedding_dim+decoder_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, trg: Tensor, hidden_state: Tensor, encoder_outputs: Tensor) -> Tensor:
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        trg = trg.unsqueeze(0)
        trg_embedded = self.dropout(self.embedding(trg))

        a = self.attention(hidden_state, encoder_outputs)

        weighted = torch.bmm(a, encoder_outputs)

        # Permutate the tensor.
        weighted = rearrange(weighted, "b a e2 -> a b e2")

        rnn_input = torch.cat((trg_embedded, weighted), dim = 2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        trg_embedded = trg_embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        logits = self.fc_out(torch.cat((output, weighted, trg_embedded), dim = 1))

        #prediction = [batch size, output dim]

        return logits, hidden.squeeze(0)


class NeuralMachineReader(nn.Module):

    def __init__(self, embedding_dim: int, encoder_dim: int, decoder_dim: int, output_dim: int,        backbone: Optional[str] = None,
            backbone_args: Optional[Dict] = None,         patch_size: Tuple[int, int] = (28, 28),
                    stride: Tuple[int, int] = (1, 14), dropout_rate: float = 0.1, teacher_forcing_ratio: float = 0.5) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.sliding_window = self._configure_sliding_window()

        self.backbone =
        self.encoder = Encoder(embedding_dim, encoder_dim, decoder_dim, dropout_rate)
        self.decoder = Decoder(embedding_dim, encoder_dim, decoder_dim, output_dim, dropout_rate)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def _configure_sliding_window(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Unfold(kernel_size=self.patch_size, stride=self.stride),
            Rearrange(
                "b (c h w) t -> b t c h w",
                h=self.patch_size[0],
                w=self.patch_size[1],
                c=1,
            ),
        )

    def forward(self, x: Tensor, trg: Tensor) -> Tensor:
        #x = [batch size, height, width]
        #trg = [trg len, batch size]

        # Create image patches with a sliding window kernel.
        x = self.sliding_window(x)

        # Rearrange from a sequence of patches for feedforward network.
        b, t = x.shape[:2]
        x = rearrange(x, "b t c h w -> (b t) c h w", b=b, t=t)

        x = self.backbone(x)
        x = rearrange(x, "(b t) h -> t b h", b=b, t=t)
