

from cs336_basics import *

def trainV1():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V1',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 1e-5, 
    lr_max = 1e-4,
    T_w = 1000,
    T_c = 10000,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 8,
    save_interval = 2000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'mps'
    )

def trainV2():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V2',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 4e-5, 
    lr_max = 2e-4,
    T_w = 20000,
    T_c = 100000,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 8,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


def trainV3():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V3',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 2e-5, 
    lr_max = 5e-4,
    T_w = 5000,
    T_c = 75000,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 16,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


def trainV4():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V4',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 5e-5, 
    lr_max = 1e-3,
    T_w = 5000,
    T_c = 75000,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 16,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


def trainV5():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V5',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 5e-5, 
    lr_max = 5e-3,
    T_w = 5000,
    T_c = 75000,
    weight_decay = 0.1, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 16,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


def trainV6():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V6',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 5e-4, 
    lr_max = 2e-3,
    T_w = 5000,
    T_c = 75000,
    weight_decay = 0.01, 
    betas = (0.9, 0.99), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 16,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


def trainV7():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/V7',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 5e-4, 
    lr_max = 2e-3,
    T_w = 5000,
    T_c = 75000,
    weight_decay = 0.01, 
    betas = (0.9, 0.95), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 16,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


# varying batch size
def trainB2():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/B2',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 5e-4, 
    lr_max = 2e-3,
    T_w = 1250,
    T_c = 18750,
    weight_decay = 0.01, 
    betas = (0.9, 0.98), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 64,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


# varying batch size
def trainB2L():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/B2L',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 1e-3, 
    lr_max = 6e-3,
    T_w = 1250,
    T_c = 18750,
    weight_decay = 0.01, 
    betas = (0.9, 0.98), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 64,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )


# varying batch size # current best
def trainB3():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/B3',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 8e-5, 
    lr_max = 4e-3,
    T_w = 1250,
    T_c = 18750,
    weight_decay = 0.01, 
    betas = (0.9, 0.98), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 32,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda'
    )



# varying batch size # current best
def trainPL1():
    train(
    enc_input_path = 'tinystories_train_encoded.npy',
    ckpt_folder = './tinystories/PL1',

    # model
    vocab_size = 10000,
    context_length = 256,
    num_layers = 4,
    dim = 512,
    num_heads = 16,
    d_ff = 2048,
    attn_pdrop = None, 
    residual_pdrop = None,

    # optimizer
    lr_min = 6e-4, 
    lr_max = 4e-3,
    T_w = 2500,
    T_c = 37500,
    weight_decay = 0.01, 
    betas = (0.9, 0.98), 
    eps = 1e-8,
    
    # training setting:
    load_ckpt = None,
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 1000,
    batch_size = 32,
    save_interval = 100000,
    max_grad_l2norm = None,
    proc_token_limit=327_680_000,
    device = 'cuda',
    model_type = TransformerLM_ParallelLayers
    )

if __name__ == "__main__":
    trainPL1()