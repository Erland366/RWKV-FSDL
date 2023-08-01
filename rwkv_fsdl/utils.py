import random
from typing import Callable, Dict, Optional

import IPython.display
import torch
import torchview
import wget
from tokenizers import Tokenizer

from rwkv_fsdl.config import CFG
from rwkv_fsdl.types import OneHot


def remap_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Renames the keys of the weights from the official implementation for clarity."""
    weights = dict(weights)  # drop "ordered" property of dict
    keys = list(weights.keys())  # pull keys out so we can mutate the dict

    for key in keys:
        value = weights.pop(key)  # pull out the tensor and drop its name from the dict
        # and then change the name
        key = (
            key
            # unabbreviate some names
            .replace("emb.", "embedding.")
            .replace(".att.", ".attention.")
            # make some names clearer
            .replace("first", "log_gain")
            .replace("decay", "log_decay")
            # make some names more precise
            .replace(".ffn.", ".gated_mlp.")
            # unpack some parameters into modules, with unabbreviated names
            .replace("mix_r", "receptance_mixer.weight")
            .replace("mix_v", "value_mixer.weight")
            .replace("gated_mlp.time_mix_k", "gated_mlp.mlp_mixer.weight")
            .replace("attention.time_mix_k", "attention.key_mixer.weight")
            .replace("log_gain", "memory.log_gain")
            .replace("log_decay", "memory.log_decay")
            # rename the embedding and unembedding layers
            .replace("blocks.0.ln0", "normalize_emb")
            .replace("ln_out", "normalize_unemb")
            .replace("head.", "unembedding.")
            # drop the "time_" prefix
            .replace(".time_", ".")
        )

        # handle different transpose convention
        if key == "embedding.weight":
            value = value.T

        # reinsert name and tensor pair
        weights[key] = value

    return weights


def prep_weights(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for key in weights.keys():
        if ".time_" in key:
            weights[key] = weights[key].squeeze()
        weights[key] = weights[key].float()
    weights = remap_weights(weights)
    return weights


class LoadingMixin:
    @classmethod
    def from_weights(cls, weights, *args, **kwargs):
        self = cls(*args, **kwargs)
        missing_keys, unexpected_keys = self.load_state_dict(weights, strict=False)
        assert not missing_keys, missing_keys
        return self


def make_graph(
    module,
    input_data: torch.Tensor,
    depth: int = 1,
    graph_dir: str = "LR",
) -> torchview.computation_graph.ComputationGraph:
    """Creates a visualizable compute graph for a module run on given input data."""

    g = torchview.draw_graph(
        module,
        input_data=input_data,
        depth=depth,
        device="meta",
        expand_nested=True,
        graph_dir=graph_dir,
        hide_module_functions=False,
    )

    return g


def display_graph(
    g: torchview.computation_graph.ComputationGraph,
    format="png",
):
    """Displays a compute graph as an image."""
    graph = g.visual_graph
    if format == "png":
        IPython.display.display_png(graph)
    elif format == "svg":
        IPython.display.display_svg(graph)
    else:
        raise ValueError("format must be 'png' or 'svg'")


def streaming_print(text: str):
    """Prints text immediately and without adding a newline."""
    print(text, end="", flush=True)


def sample(probs: torch.Tensor, temperature=1.0, top_p=1.0) -> int:
    """Samples an index from a probability distribution."""
    if temperature == 0:
        return torch.argmax(probs).item()

    probs = probs.clone()  # work on a copy, rather than the original
    if top_p < 1.0:
        probs = restrict_to_top_p(probs, top_p=top_p)  # aka nucleus sampling

    probs = probs ** (1 / temperature)  # apply temperature scaling

    probs = probs / torch.sum(probs)  # renormalize

    nxt: int = random.choices(range(len(probs)), weights=probs, k=1)[0]

    return nxt


def restrict_to_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sets all but the top p probability mass to zero, as used in nucleus sampling."""
    sorted_probs, sort_indices = torch.sort(probs, descending=True, stable=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    above_threshold = cumulative_probs >= top_p
    all_below_threshold, first_below_threshold = torch.max(above_threshold, dim=-1)

    if not all_below_threshold:
        indices_of_bottom_p = sort_indices[first_below_threshold:]
        probs[indices_of_bottom_p] = 0.0

    return probs


# ==================================My Custom Utils===================================
def setup_tokenizer(tokenizer: Optional[Callable] = None):
    if not CFG.TOKENIZER_PATH.exists():
        print("Tokenizer file doesn't exists, Downloading it")
        print("Create necessary path for tokenizer file")
        CFG.TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _url = f"https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/\
            {CFG.TOKENIZER_PATH.name}"
        print(f"Downloading tokenizer from {_url}")
        wget.download(url=_url, out=CFG.TOKENIZER_PATH.parent.as_posix())
    print("Loading Tokenizer")
    tokenizer = Tokenizer.from_file(CFG.TOKENIZER_PATH.as_posix())
    tokenizer.token_to_id: Callable[[str], int]
    return tokenizer


def setup_weights() -> None:
    if not CFG.WEIGHTS_PATH.exists():
        print("Create necessary path")
        CFG.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _url = f"https://huggingface.co/BlinkDL/rwkv-4-pile-430m/resolve/main/\
            {CFG.WEIGHTS_PATH.name}"
        print(f"Downloading weights from {_url}")
        wget.download(
            url=_url,
            out=CFG.WEIGHTS_PATH.parent.as_posix(),  # must be directory
        )

    print(f"Loading {CFG.WEIGHTS_PATH}")
    weights = torch.load(CFG.WEIGHTS_PATH, map_location="cpu")
    weights = prep_weights(weights)
    return weights


def setup_tokenizer() -> Callable[[str], int]:
    if not CFG.TOKENIZER_PATH.exists():
        print("Create necessary path")
        CFG.TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        _url = f"https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/\
            {CFG.TOKENIZER_PATH.name}"
        print(f"Downloading tokenizer from {_url}")
        wget.download(url=_url, out=CFG.TOKENIZER_PATH.parent.as_posix())
    print("Loading Tokenizer")
    tokenizer = Tokenizer.from_file(CFG.TOKENIZER_PATH.as_posix())
    tokenizer.token_to_id: Callable[[str], int]
    return tokenizer


def testing_tokenizer(token_id: int = 50_287) -> None:
    tokenizer = setup_tokenizer()
    if 0 <= token_id < CFG.N_VOCAB:
        assert tokenizer.id_to_token(CFG.N_VOCAB) is None
        print(f"index {token_id} is in vocab")
    else:
        assert tokenizer.id_to_token(CFG.N_VOCAB) is None
        print(f"index {token_id} is not in vocab")


def to_onehot(k: int) -> OneHot:
    out = torch.zeros(CFG.N_VOCAB)
    out[k] = 1.0
    return out
