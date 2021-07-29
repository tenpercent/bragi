from .parse import parse, AMRIO
from tqdm import tqdm

def parse_amr_bank(filename: str, **parse_kwargs):

    sentences = [amr.sentence \
        for amr in \
            tqdm(AMRIO.read(filename), desc=f"Reading from {filename}")]

    return parse(sentences, **parse_kwargs)
