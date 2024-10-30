
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
    model_type: Type[torch.nn.Module] = TransformerLM
    ) -> str:

    model = model_type(vocab_size, context_length, num_layers, dim, num_heads, d_ff).to(device)
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
        "Lily and Tom were twins who liked to decorate things.",
        max_len=256,
        T = 0.6,
        p_threshold=0.95,

        model_ckpt = './tinystories/SELF1/40001.pth',
        tokenizer_file = 'tinystories_tok.json',

        device = 'cuda',
        model_type = SelfTransformer
    ))