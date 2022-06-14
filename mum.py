# mum.py
from collections import defaultdict
from argparse import ArgumentParser
import numpy as np
import numba as nb



def _fasta_reads_from_filelike(f, COMMENT=b';'[0], HEADER=b'>'[0]):
    """internal function that yields facta records as (header: bytes, seq: bytearray)"""
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


def make_genome_text(filename, sep=ord("&"), end=ord("$")):
    """
    Create a concatenated text from a genomic FASTA file,
    using the given sequence separator byte (sep) and sentinel byte (end).
    Return a bytearray with the concatenated bytes.
    """
    text = bytearray()
    with open(filename, "rb") as f:
        for (header, seq) in _fasta_reads_from_filelike(f):
            text.extend(seq)
            text.append(sep)  # the separator byte
        text.append(end)  # the end byte (sentinel)
    return text


def compute_pos_builtin(T):
    """
    using built-in sort with custom key function;
    SLOW on repetitive texts: O(n^2 log n).
    Needs a HUGE amount of memory O(n^2) because it instantiates each suffix.
    But implementing SAIS here would be too much work.
    """
    if len(T) > 10_000:
        raise RuntimeError("ERROR: Using built-in sort on texts over 10_000 characters will kill your memory!")
    suffixes = lambda p: t[p:]
    pos = sorted(range(len(t)), key=suffixes)
    # return numpy array -- for short texts, 32 bits is enough
    return np.array(pos, dtype=np.int32)


def compute_pos_manber_myers(T):
    """
    using classical Manber-Myers doubling technique.
    OK performance of O(n log n) time -- this implementation may be slower.
    """
    def sort_bucket(t, bucket, result, order=1):
        d = defaultdict(list)
        for i in bucket:
            key = t[i:i+order]
            d[key].append(i)
        for k, v in sorted(d.items()):
            if len(v) > 1:
                result = sort_bucket(t, v, result, order*2)
            else:
                result.append(v[0])
        return result
    result = sort_bucket(T, range(len(T)), [], order=1)  # Python list
    pos = np.array(result, dtype=np.int32)  # convert to numpy array
    return pos


@nb.njit(nopython=True)
def compute_lcp(T, pos):
    """
    lcp using Kasai's linear-time algorithm on numpy arrays
    """
    n = len(pos)
    lcp = np.zeros(n+1, dtype=np.int32)
    lcp[0] = lcp[n] = -1  # border sentinels
    # compute rank, the inverse of pos
    rank = np.zeros(n, dtype=np.int32)
    for r in range(n):
        rank[pos[r]] = r

    lp = 0 # current common prefix length
    for p in range(n-1):
        r = rank[p]
        if r == 0:  # pos[r] must be a sentinel, so lcp[r]=0
            lcp[r] = 0
            continue
        pleft = pos[r-1]  # r-1 is now valid
        while T[p+lp] == T[pleft + lp]:
            lp += 1
        lcp[r] = lp
        lp = lp - 1 if lp > 0 else 0  # next suffix: lose first character
    return lcp


def print_arrays(T, pos, lcp):
    for r in range(len(pos)):
        print(f"{pos[r]:2d}  {lcp[r]:2d}  {T[pos[r]:].decode('ASCII')}")


@nb.njit
def decode_bytes(x):
    return ''.join([chr(z) for z in x]) #To decode the bytes

def count_mums(T, pos, lcp, n1, minlen=0, show=False):
    """
    T: a `bytes` object containing the concatenated genomes;
    pos: the suffix array of T;
    lcp: the lcp array of T;
    n1: the length of the first genome (T[:n1] is the first genome);
    minlen: report only MUMs of length at least `minlen`;
    show: if show=True, print MUMs, otherwise just count them and their length.

    Return the number and total length of MUMs (of the given minimum length)
    """
    n = len(pos)
    T = decode_bytes(T) #decode the bytes
    n2 = len(T) - n1 #the length of the second genome
    nmum = lmum = 0  # number and total length of MUMs (of given minlen)
    mum = ""
    mums = list()
    for r in range(1,n):
        p1, p2 = pos[r-1], pos[r]
        if (p1 <= n2) and (p2 <= n2):
            continue 
        if (p1 > n2) and (p2 > n2):
            continue
        if (lcp[r-1] >= lcp[r]) or (lcp[r+1] >= lcp[r]):
            continue
        if (p1 == 0) or (p2 == 0) or (T[p1-1] != T[p2-1]):
            mum = T[p1:p1+lcp[r]] #save th MUM
            if len(mum) > minlen: #if length of the MUM is greater than minlen
                mums.append(mum) #append it to MUMs
                nmum  = nmum + 1  #increase the number of MUMs
                lmum = lmum + len(mum) #add the length to lmum
    if(show == True): 
        return mums #print the mums if show = True
    else:
        return nmum, lmum  # number and total length of MUMs


def get_argument_parser():
    p = ArgumentParser(description="finds all Maximal Unique Matches (MUMs) between two genomes")
    p.add_argument("fasta1",
        help="name of first FASTA file: first genome")
    p.add_argument("fasta2",
        help="name of second FASTA file: second genome")
    p.add_argument("--minlen", "-m", type=int, default=0,
        help="minimum length of MUMs to consider (default=0; use >= 16 for bacterial genomes)")
    p.add_argument("--show", action="store_true",
        help="print MUMs to stdout")
    return p


def main(args):
    print(f"# Reading '{args.fasta1}'...")
    T = make_genome_text(args.fasta1, sep=ord("&"), end=ord("$"))
    print(f"# Reading '{args.fasta2}'...")
    S = make_genome_text(args.fasta2, sep=ord("%"), end=ord("#"))
    n1, n2 = len(T), len(S)  # lengths of the individual genomes
    T = bytes(T+S)
    n = len(T)
    print(f"# Genome lengths: {n1} + {n2} = {n}")
    print(f"# Computing suffix array...")
    pos = compute_pos_manber_myers(T)
    print(f"# Computing lcp array...")
    lcp = compute_lcp(T, pos)
    if n <= 50: print_arrays(T, pos, lcp)  # only actually prints short texts

    # search for MUMs and count / print them
    print(f"# Looking for MUMs...")
    nmums, lmums = count_mums(T, pos, lcp, n1, minlen=args.minlen, show=args.show)
    print(f"# Found {nmums} MUMs of total length {lmums}.")
    print(f"# Done.")


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)
