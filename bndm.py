import argparse  # for command line interface
import numpy as np  # for typed arrays
from numba import njit, uint64, uint8 # for just-in-time compilation

def _fasta_reads_from_filelike(f, COMMENT=b';'[0], HEADER=b'>'[0]):
    """internal function that yields fasta records as (header: bytes, seq: bytearray)"""
    strip = bytes.strip
    header = seq = None
    for line in f:
        line = strip(line)
        if len(line) == 0:
            continue
        if line[0] == COMMENT:
            continue
        if line[0] == HEADER:
            if header is not None:
                yield (header, seq)
            header = line[1:]
            seq = bytearray()
            continue
        seq.extend(line)
    if header is not None:
        yield (header, seq)


def fasta_items(filename):
    """
    generator function that yields each (header, sequence) pair from a FASTA file.
    Header is given as an immutable 'bytes' object;
    sequence is given as a mutable numpy array of dtype uint8.
    """
    with open(filename, "rb") as f:
        for (header, seq) in _fasta_reads_from_filelike(f):
            yield (header, np.frombuffer(seq, dtype=np.uint8))

# build nfas

@njit
def build_nfa_and(P):
    """Build an NFA from a pattern"""
    mask = np.zeros(256, dtype=np.uint64)  # one mask for each byte 0..255
    for bit, a in enumerate(P):
        mask[a] |= uint64(1 << bit)
    accept = (1 << (len(P)-1))
    return mask, accept


@njit
def build_nfa_or(P):
    """Build an "inverse" NFA from a pattern"""
    # first, build the Shift_And NFA and then modify it
    mask, accept = build_nfa_and(P)
    # invert all masks, except accept
    for a in range(256):
        mask[a] = uint64(~mask[a])
    return mask, accept

# matchings

@njit(locals=dict(k=uint64, A=uint64, p=uint64, c=uint8))
def shift_and(mask, accept, text, results):
    """
    just-in-time compiled version of the Shift-And matcher
    that writes end positions of matches into an array 'results'
    and returns the number of matches.
    """
    k=0
    N = results.size
    A = 0
    for p, c in enumerate(text):
        A = ((A << 1) | 1) & mask[c]
        if A & accept:  # test whether accept bit is nonzero
            if k < N: results[k] = p
            k += 1
    return k


@njit(locals=dict(k=uint64, A=uint64, p=uint64, c=uint8))
def shift_or(mask, accept, text, results):
    """
    just-in-time compiled version of the Shift-Or matcher
    that writes end positions of matches into an array 'results'
    and returns the number of matches.
    """
    k = 0
    N = results.size
    A = uint64(-1)  # all bits set
    for p, c in enumerate(text):
        A = (A << 1) | mask[c]
        if A & accept == 0:  # test whether accept bit is zero
            if k < N: results[k] = p
            k += 1
    return k

@njit(locals=dict(accept_state=uint64,k=uint64,m=uint64,n=uint64,pos=uint64))
    
def bndm(masks, accept_state, T, results):
    """
    Input:
    masks: Array with 256 slots of uint64
    accept_state: Bit mask where exactly one bit is 1. This describes the accepting state
    T: Text in ASCII encoding
    resultus: Numpy array to store results
    Output:
    k: Number of matchings
    """
    k = 0
    N = results.size
    pattern_length = int(np.log2(accept_state)+1)
    n, m, pos = len(T), pattern_length, pattern_length
    while(pos <= n):
        j, lsuff, D = 1, 0, (1 << m) - 1
        while (D != 0):
            D &= masks[T[pos-j]]
            if D & accept_state != 0:
                if j == m:
                    if k < N:
                        results[k] = pos - 1
                        k+=1 #increase number of matchings
                    break
                else:
                    lsuff = j
            j+=1
            D = D << 1
        pos += m -lsuff
    return k

def main(args):
    alg = args.algorithm
    P = args.pattern.encode("ASCII")

    if alg == "and":
        build_nfa = build_nfa_and
        find_matches = shift_and
    elif alg == "or":
        build_nfa = build_nfa_or
        find_matches = shift_or
    elif alg == "bndm":
        build_nfa = lambda x: build_nfa_and(x[::-1])
        find_matches = bndm
    NRESULTS = args.maxresults
    results = np.zeros(NRESULTS, dtype=np.uint64)

    nfa = build_nfa(P)

    for header, sequence in fasta_items(args.fasta):
        print("#", header.decode("ASCII"))
        nresults = find_matches(*nfa, sequence, results)
        if nresults > NRESULTS:
            print("! Too many results, showing first {NRESULTS}")
            nresults = NRESULTS
        print(*list(results[:nresults]), sep="\n")


def get_argument_parser():
    p = argparse.ArgumentParser(description="Pattern search, shift_and, shift_or, bndm")
    p.add_argument("--fasta", "-f", required=True,
        help="FASTA file of genome")
    p.add_argument("-P", "--pattern",
        help="immediate pattern to be matched")
    p.add_argument("-a", "--algorithm", metavar="ALGORITHM",
        default="and", choices=("and", "or", "bndm"),
        help="algorithm to use ('and' (default), 'or', 'bndm')")
    p.add_argument("--maxresults", "-R", type=int, default=10_000,
        help="maximum number of results to output [10_000]")
    return p


if __name__ == "__main__":
    main(get_argument_parser().parse_args())
