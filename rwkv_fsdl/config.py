from pathlib import Path


class CFG:
    N_VOCAB = 50_277
    WEIGHTS_PATH = Path.cwd() / "files" / "RWKV-4-Pile-430M-20220808-8066.pth"
    TOKENIZER_PATH = Path.cwd() / "files" / "20B_tokenizer.json"
    N_EMBD = 1024
    MLP_HIDDEN_DIM = 4 * N_EMBD
