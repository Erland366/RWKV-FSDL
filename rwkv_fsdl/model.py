import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from typing import Callable, Tuple

from tokenizers import Tokenizer

from rwkv_fsdl import utils
from rwkv_fsdl.utils import LoadingMixin, to_onehot, setup_weights
from rwkv_fsdl.types import (
    NextTokenProbabilities,
    ScalingWeight,
    TokenId,
    Embedding,
    Update,
    ChannelParameter,
    Latents,
)
from rwkv_fsdl.config import CFG

torch.autograd.set_grad_enabled(False)
torch.set_default_dtype(
    torch.float64
)  # Because of the implementation with exp, for now use more precision


class RWKV(LoadingMixin, nn.Module):
    def __init__(self, rwkv_blocks: list):
        super().__init__()
        self.blocks = nn.Sequential(*rwkv_blocks)

    @beartype
    def forward(self, x: Embedding) -> Embedding:
        for ii, block in enumerate(self.blocks):
            x = block(x)
        return x


class RWKVBlock(nn.Module):
    """The core block in the RWKV architecture, which updates the embedding"""

    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(CFG.N_EMBD)
        self.attention = AttentionBlock()

        self.ln2 = nn.LayerNorm(CFG.N_EMBD)
        self.gated_mlp = GatedMLP()

    @beartype
    def forward(self, x: Embedding) -> Embedding:
        # attention enriches embedding using sequence memory
        dx: Update = self.attention(self.ln1(x))
        x: Embedding = x + dx  # residual network

        # gated MLP enriches embedding by doing computations
        dx: Update = self.gated_mlp(self.ln2(x))
        x: Embedding = x + dx  # residual network

        return x


class Mixer(LoadingMixin, nn.Module):
    """Returns a per-entry-weighted combination of current input and previous input"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(CFG.N_EMBD))
        self.register_buffer(
            "last_x", torch.zeros(CFG.N_EMBD), persistent=False
        )  # persistent=False means "don't save to disk"

    @beartype
    def forward(self, current_x: Embedding) -> Embedding:
        out = mix_embeddings(current_x, self.last_x, self.weight)
        self.last_x: Embedding = current_x  # store for later
        return out


@beartype
def mix_embeddings(
    x: Embedding, y: Embedding, mixing_params: ChannelParameter
) -> Embedding:
    """Mixes two embeddings with weights given by the mixing_params"""
    return x * mixing_params + y * (1 - mixing_params)


class SquaredReLU(nn.Module):
    def forward(self, x: Latents) -> Latents:
        return F.relu(x) ** 2


class GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # non standard terminology of RWKV where key is first layer of MLP, value is second
        self.key = nn.Linear(CFG.N_EMBD, CFG.MLP_HIDDEN_DIM, bias=False)
        self.nonlinearity = SquaredReLU()
        self.value = nn.Linear(CFG.MLP_HIDDEN_DIM, CFG.N_EMBD, bias=False)

        self.mlp_mixer, self.receptance_mixer = Mixer(), Mixer()
        self.receptance = nn.Linear(CFG.N_EMBD, CFG.N_EMBD, bias=False)

    @beartype
    def forward(self, x: Embedding) -> Update:
        # "mix" current input with the previous input
        mixed_x: Embedding = self.mlp_mixer(x)
        # put that through an MLP
        mlp_outputs: Embedding = self.value(self.nonlinearity(self.key(mixed_x)))
        # non-standard nomenclature, probably because of this paper https://arxiv.org/abs/2012.14913

        # "mix" the current input with the previous input again, with different weights
        mixed_x_receptance: Embedding = self.receptance_mixer(x)
        # use that to calculate how "receptive" each dimension of embedding is to new inputs
        receptance: Embedding = self.receptance(mixed_x_receptance)

        # Convert that receptance to a 0-1 value with a sigmoid
        gating_values: Embedding = torch.sigmoid(receptance)
        # Then use those as "gating" by multiplying them
        dx: Update = gating_values * mlp_outputs

        return dx


class AttentionBlock(nn.Identity):
    def __init__(self):
        super().__init__()

        # rwkv
        self.key = nn.Linear(CFG.N_EMBD, CFG.N_EMBD, bias=False)
        self.value = nn.Linear(CFG.N_EMBD, CFG.N_EMBD, bias=False)
        self.receptance = nn.Linear(CFG.N_EMBD, CFG.N_EMBD, bias=False)
        self.output = nn.Linear(CFG.N_EMBD, CFG.N_EMBD, bias=False)

        # mixers
        self.key_mixer, self.value_mixer = Mixer(), Mixer()
        self.receptance_mixer = Mixer()

        # memory
        self.memory: nn.Module = WKVMemory()

    @beartype
    def forward(self, x: Embedding) -> Update:
        # as with the MLP, do mixers before anything else
        mixed_keys = self.key_mixer(x)
        keys: Embedding = self.key(mixed_keys)

        mixed_values = self.value_mixer(x)
        values: Embedding = self.value(mixed_values)

        # wkv: apply weighted decay to merge
        #      current info (k and v) with past
        wkv: Embedding = self.memory(values, torch.exp(keys))

        # decide how r each channel is to inputs
        mixed_receptances = self.receptance_mixer(x)
        receptances = self.receptance(mixed_receptances)
        gating_values = torch.sigmoid(receptances)

        # rwkv : use the r to gate the output of the "wkv" memory
        rwkv: Embedding = gating_values * wkv

        # and then do one final linear transform before returning it
        dx: Update = self.output(rwkv)

        return dx


class WKVMemory(nn.Module):
    """A memory module whose contents exponentially decay over time, at a different rate per channel
    This is the "Positional Embedding"
    """

    def __init__(self):
        super().__init__()

        # Learned memory parameters -- one value for each dimension in the embeddings
        self.log_gain: ChannelParameter = nn.Parameter(torch.zeros(CFG.N_EMBD))
        self.log_decay: ChannelParameter = nn.Parameter(torch.zeros(CFG.N_EMBD))

        # state buffers to track information accros a sequence
        contents, normalizer = torch.zeros(CFG.N_EMBD), torch.zeros(CFG.N_EMBD)
        self.register_buffer("contents", contents, persistent=False)
        self.register_buffer("normalizer", normalizer, persistent=False)

    def update(
        self, importances: ScalingWeight, values: Embedding
    ) -> Tuple[Update, Update]:
        """Updates the memory by incrementing time and mixing in the weighted input values."""
        # decay the information currently in memory by one step
        self.step()

        # compute new information to add to the memory
        contents_update: Update = (
            importances * values
        )  # Scale each value by the matching importance weight
        normalizer_update: Update = (
            importances  # Keep track of the weights so we can normalize across steps
        )

        # and then add the new information to the memory
        self.contents += contents_update
        self.normalizer += normalizer_update  # -- including updating the normalizer

        # and return it
        return contents_update, normalizer_update

    def step(self):
        """Pushes the information currently in the memory towards zero."""
        decay_rate: ScalingWeight = torch.exp(
            self.log_decay
        )  # exp ensures that decay rate is positive
        self.contents *= torch.exp(
            -decay_rate
        )  # decay rate > 0, so exp(-decay_rate) < 1
        self.normalizer *= torch.exp(
            -decay_rate
        )  # so each .step shrinks the contents and normalizer towards 0

    def apply_gain(self, latest_contents, latest_normalizer):
        """Applies the channelwise gain to the latest contents and normalizer"""
        gain = torch.exp(self.log_gain) - 1  # -1 < gain < inf

        boosted_contents = gain * latest_contents
        boosted_normalizer = gain * latest_normalizer
        return boosted_contents, boosted_normalizer

    @beartype
    def forward(self, values: Embedding, importances: ScalingWeight) -> Update:
        """Applies the RWKV "time-mixing block" forward pass, in the "RNN Cell" style"""

        # first, we update the memory and return what we just added
        latest_contents, latest_normalizer = self.update(importances, values)

        # then we adjust the representation of the latest information
        latest_contents, latest_normalizer = self.apply_gain(
            latest_contents, latest_normalizer
        )

        # before adding it in and dividing, to get the final thing we report as output
        out: Update = (self.contents + latest_contents) / (
            self.normalizer + latest_normalizer
        )

        return out


class AutoRegressiveLM(nn.Module):
    """An LM that can continue a sequence by generating one token at a time."""

    def __init__(self, embedding_layer, blocks, unembedding_layer):
        super().__init__()
        self.tokenizer = Tokenizer.from_file(CFG.TOKENIZER_PATH.as_posix())
        self.embedding: TokenEmbedding = embedding_layer
        self.blocks: Callable[[Embedding], Embedding] = blocks  # rwkv is here!
        self.unembedding: Unembedding = unembedding_layer

    @beartype
    def forward(self, token: TokenId) -> NextTokenProbabilities:
        token = to_onehot(token)

        # use that onehot to retrieve the token's dense vector representation, or "embedding"
        embedded_token: Embedding = self.embedding(token)

        # apply the meat of the model to enrich the embedding
        sequence_embedding: Embedding = self.blocks(embedded_token)

        # use that to assign probabilities to each possible next token
        probs: NextTokenProbabilities = self.unembedding(sequence_embedding)

        return probs

    @beartype
    def generate(
        self,
        sequence: str = "",
        N: int = 1,
        temperature: float = 0.1,
        top_p: float = 1.0,
    ) -> NextTokenProbabilities:
        """Generates N additional tokens that might follow the provided sequence."""

        token_ids = self.tokenizer.encode(sequence).ids

        if not (sequence_length := len(token_ids)):  # handle empty sequence
            probs: NextTokenProbabilities = self(
                0
            )  # 0 is a special token id, marks a boundary

        for ii in range(sequence_length + N):
            if ii < sequence_length:  # at first, tokens come from the sequence
                token = token_ids[ii]
            else:  # then after that, we're generating new tokens
                token = utils.sample(probs, temperature=temperature, top_p=top_p)

            # We get the probabilities for the next token by calling the model on the current token
            probs: NextTokenProbabilities = self(token)

            # and print the sequence as we go
            utils.streaming_print(self.tokenizer.decode([token]))

        return probs


class TokenEmbedding(LoadingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(
            in_features=CFG.N_VOCAB, out_features=CFG.N_EMBD, bias=False
        )
        self.normalize_emb = nn.LayerNorm(CFG.N_EMBD)

    @beartype
    def forward(self, token: TokenId) -> Embedding:
        token = to_onehot(token) if isinstance(token, int) else token
        embedded_token: Embedding = self.embedding(token)
        normalized_embedded_token = self.normalize_emb(embedded_token)

        return normalized_embedded_token


class Unembedding(LoadingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize_unemb = nn.LayerNorm(CFG.N_EMBD)
        self.unembedding = nn.Linear(CFG.N_EMBD, CFG.N_VOCAB, bias=False)

    @beartype
    def forward(self, x: Embedding) -> NextTokenProbabilities:
        normalized_embedding = self.normalize_unemb(x)
        logits = self.unembedding(normalized_embedding)

        # we convert them to probabilities with the softmax function
        probs: NextTokenProbabilities = nn.functional.softmax(logits, dim=-1)

        return probs


def clear_buffers(module, verbose=False):
    for name, buffer in module.named_buffers():
        if verbose:
            print(f"clearing buffer {name}")
        buffer.zero_()


AutoRegressiveLM.clear_buffers = clear_buffers
RWKV.clear_buffers = clear_buffers
RWKVBlock.clear_buffers = clear_buffers


