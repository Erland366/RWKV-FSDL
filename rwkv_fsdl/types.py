import torch
from jaxtyping import Float64

from rwkv_fsdl.config import CFG

Float = Float64
OneHot = Float[torch.Tensor, f"vocabSize={CFG.N_VOCAB}"]
TokenId = int | OneHot
NextTokenProbabilities = Float[torch.Tensor, f"vocabSize={CFG.N_VOCAB}"]
Embedding = Float[torch.Tensor, f"channels={CFG.N_EMBD}"]
ChannelParameter = Float[torch.Tensor, f"params={CFG.N_EMBD}"]
Latents = Float[torch.Tensor, f"latents={CFG.MLP_HIDDEN_DIM}"]
ScalingWeight = Float[
    torch.Tensor, f"positiveEntries={CFG.N_EMBD}"
]  # positive number, one per channel
Update = Embedding
