
from cs336_basics import *

def gen_test(
    prompt: str,
    max_len: int,
    T: float,
    p_threshold: float,

    model_ckpt: str,
    tokenizer_file: str,

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    device = 'mps',
    ) -> str:

    model = TransformerLM(vocab_size, context_length, num_layers, dim, num_heads, d_ff).to(device)
    tokenizer = BPETok.from_local(tokenizer_file)
    
    obj = torch.load(model_ckpt, weights_only=True)
    model.load_state_dict(obj['model_dict'])
    model.eval()
    model.device = device

    return decode(model,tokenizer,
           prompt,
           max_len, T, p_threshold)


if __name__ == "__main__":
    print(gen_test(
        "Alice used to have a cat.",
        max_len=256,
        T = 0.3,
        p_threshold=0.95,

        model_ckpt = './tinystories/V3/80001.pth',
        tokenizer_file = 'tinystories_tok.json',

        device = 'cuda'
    ))