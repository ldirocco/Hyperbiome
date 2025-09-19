import pandas as pd
import pickle
import re
import struct
from typing import List

def map_species_to_id(seen_gallery_path, unseen_gallery_path):
    seen_gallery=pd.read_csv(seen_gallery_path, sep='\t')
    seen_species=seen_gallery["Species"].unique()
    species_to_id= {species: i for i, species in enumerate(seen_species)}

    last_id=max(species_to_id.values())+1
    unseen_gallery=pd.read_csv(unseen_gallery_path, sep='\t')
    unseen_species=unseen_gallery["Species"].unique()

    for species in unseen_species:
        species_to_id[species] = last_id
        last_id += 1

    with open("species_dictionary", 'wb') as f:
        pickle.dump(species_to_id, f)

def map_genus_to_id(seen_gallery_path, unseen_gallery_path):
    seen_gallery=pd.read_csv(seen_gallery_path, sep='\t')
    seen_species=seen_gallery["Species"].unique()

    seen_genus=set(
        re.sub(r"[-_][A-Z]+$", "", species.split()[0])
        for species in seen_species
    )
    genus_to_id={genus:i for i, genus in enumerate(seen_genus)}

    next_id=max(genus_to_id.values())+1

    unseen_gallery=pd.read_csv(unseen_gallery_path, sep='\t')
    unseen_species=unseen_gallery["Species"].unique()
    unseen_genus=set(
        re.sub(r"[-_][A-Z]+$", "", species.split()[0])
        for species in unseen_species
    )

    for genus in unseen_genus:
        if genus not in genus_to_id:
            genus_to_id[genus] = next_id
            next_id += 1
    with open("genus_dictionary", 'wb') as f:
        pickle.dump(genus_to_id, f)

def decompress_hd_sketch(sketch: dict) -> List[int]:
    hv_d = sketch["hv_d"]
    quant_bit = sketch["hv_quant_bits"]
    hv_packed = sketch["hv"]  # list of int16 or int (compressed)

    hv_decompressed = [0] * hv_d

    for i in range(quant_bit * hv_d):
        # Estrai il bit i
        bit_val = (hv_packed[i // 16] >> (i % 16)) & 1
        # Scrivilo nel giusto posto
        hv_decompressed[i // quant_bit] |= bit_val << (i % quant_bit)

        # Quando hai riempito quant_bit per un numero, converti in signed int
        if (i + 1) % quant_bit == 0:
            val = hv_decompressed[i // quant_bit]
            if val > (1 << (quant_bit - 1)):
                hv_decompressed[i // quant_bit] = val - (1 << quant_bit)

    return hv_decompressed

def read_u8(f):
    return struct.unpack('<B', f.read(1))[0]

def read_u64(f):
    return struct.unpack('<Q', f.read(8))[0]

def read_bool(f):
    val = f.read(1)
    return val != b'\x00'

def read_i32(f):
    return struct.unpack('<i', f.read(4))[0]

def read_usize(f):
    # Assume 64-bit usize
    return struct.unpack('<Q', f.read(8))[0]

def read_string(f):
    length = read_u64(f)
    return f.read(length).decode('utf-8')

def read_vec_i16(f):
    length = read_u64(f)
    # i16 = 2 bytes
    data = f.read(length * 2)
    return list(struct.unpack(f'<{length}h', data))

def read_sketch(f):
    ksize = read_u8(f)
    scaled = read_u64(f)
    canonical = read_bool(f)
    seed = read_u64(f)
    hv_d = read_usize(f)
    hv_quant_bits = read_u8(f)
    hv_norm_2 = read_i32(f)
    file_str = read_string(f)
    hv = read_vec_i16(f)

    return {
        'ksize': ksize,
        'scaled': scaled,
        'canonical': canonical,
        'seed': seed,
        'hv_d': hv_d,
        'hv_quant_bits': hv_quant_bits,
        'hv_norm_2': hv_norm_2,
        'file_str': file_str,
        'hv': hv,
    }


def read_file_sketch(f):
    length = read_u64(f)
    return [read_sketch(f) for _ in range(length)]
