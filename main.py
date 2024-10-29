

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
    device = 'mps'
    )

def trainV2():
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
    device = 'mps'
    )

if __name__ == "__main__":
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
    load_ckpt = './tinystories/V1/103146.pth',
    valid_enc_input = 'tinystories_valid_encoded.npy',
    valid_interval = 500,
    batch_size = 8,
    save_interval = 10000,
    max_grad_l2norm = None,
    device = 'mps'
    )