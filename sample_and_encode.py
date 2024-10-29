from cs336_basics import *
import regex as re
import numpy as np

import random
import cProfile

def chunk(text: str) -> list[str]:
    PAT = r"\S[\s\S]*?<\|endoftext\|>"
    return re.findall(PAT, text)

def test():
    with open("data/TinyStoriesV2-GPT4-valid.txt", 'r') as f:
        text = f.read(1024*1024)

    # with open("data/owt_valid.txt", 'r') as f:
    #     text = f.read()
    
    chunks = chunk(text)

    samples = random.choices(chunks, k=50)

    samples = chunks

    samples = [text]

    # calculate the total byte size of the chunks
    total_size = sum(len(chunk.encode('utf-8')) for chunk in samples)

    tokenizer = BPETok.from_local('tinystories_tok.json')

    start_time = time.time()
    # encode the samples
    encoded_samples = [tokenizer.encode(sample) for sample in samples]
    end_time = time.time()


    # calculate the total length of the encoded samples
    total_length = sum(len(sample) for sample in encoded_samples)

    # calculate the compression ratio
    compression_ratio = total_size / total_length

    print(f"Total byte size of the chunks: {total_size}")
    print(f"Total length of the encoded samples: {total_length}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Time taken to encode the samples: {end_time - start_time} seconds")
    print(f"Average throughput: {total_size / (end_time - start_time)} bytes per second")

def encode_tinystories():
    with open("data/TinyStoriesV2-GPT4-train.txt", 'r') as f:
        text = f.read()

    tokenizer = BPETok.from_local('tinystories_tok.json')

    total_size = len(text.encode('utf-8'))

    start_time = time.time()
    # encode the samples
    encoding_result = tokenizer.encode(text)
    end_time = time.time()

    # calculate the total length of the encoded samples
    total_length = len(encoding_result)

    # calculate the compression ratio
    compression_ratio = total_size / total_length

    print(f"Total byte size of the chunks: {total_size}")
    print(f"Total length of the encoded samples: {total_length}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Time taken to encode the samples: {end_time - start_time} seconds")
    print(f"Average throughput: {total_size / (end_time - start_time)} bytes per second")

    ids = np.array(encoding_result, dtype=np.uint16)
    
    # serialize the encoding result
    np.save("tinystories_encoded.npy", ids)



def encode_owt():
    with open("data/TinyStoriesV2-GPT4-valid.txt", 'r') as f:
        text = f.read()

    tokenizer = BPETok.from_local('tinystories_tok.json')

    total_size = len(text.encode('utf-8'))

    start_time = time.time()
    # encode the samples
    encoding_result = tokenizer.encode(text)
    end_time = time.time()

    # calculate the total length of the encoded samples
    total_length = len(encoding_result)

    # calculate the compression ratio
    compression_ratio = total_size / total_length

    print(f"Total byte size of the chunks: {total_size}")
    print(f"Total length of the encoded samples: {total_length}")
    print(f"Compression ratio: {compression_ratio}")
    print(f"Time taken to encode the samples: {end_time - start_time} seconds")
    print(f"Average throughput: {total_size / (end_time - start_time)} bytes per second")

    ids = np.array(encoding_result, dtype=np.uint16)
    
    # serialize the encoding result
    np.save("tinystories_valid_encoded.npy", ids)

if __name__ == '__main__':
    encode_owt()
    # cProfile.run("test()", "profile.txt")

    # import pstats
    # p = pstats.Stats("profile.txt")

    # p.strip_dirs().sort_stats('tottime').print_stats(50)

