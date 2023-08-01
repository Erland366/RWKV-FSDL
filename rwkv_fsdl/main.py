
from rwkv_fsdl.model import RWKV, AutoRegressiveLM, RWKVBlock, TokenEmbedding, Unembedding
from rwkv_fsdl.utils import setup_tokenizer, setup_weights


def main() -> None:
    setup_tokenizer() # to download the tokenizer
    weights = setup_weights()
    embs = TokenEmbedding.from_weights(weights=weights)
    unembs = Unembedding.from_weights(weights=weights)
    N_LAYER = 24
    rwkv_blocks = [RWKVBlock() for _ in range(N_LAYER)]

    rwkv = RWKV.from_weights(weights, rwkv_blocks)

    rwkv4 = AutoRegressiveLM(embs, rwkv, unembs)
    rwkv4 = rwkv4.eval()

    rwkv4.clear_buffers()

    rwkv4.generate(sequence="Drosophila", N=8, temperature=0.0)


if __name__ == "__main__":
    main()