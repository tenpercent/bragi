from .parse import parse, AMRIO

def parse_amr_bank(filename: str, **parse_kwargs):
    amrs_in = list(AMRIO.read(filename))

    sentences = [amr.sentence for amr in amrs_in]

    return parse(sentences, **parse_kwargs)


