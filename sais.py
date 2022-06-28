from collections import Counter
import numpy as np
from numba import njit

SMALLER, LARGER = 0, 1  # type constants

# We assume that the text T is an object of type 'bytes', 'bytearray',
# or 'numpy.array' of dtype uint8, or a larger numeric type (like uint64),
# but not a Python unicode string.
# Therefore, it is initially converted from input str to uint8 (ASCII).
# We always maintain the invariant that the sentinel is represented by 0.


@njit
def count_cumulative_characters(T, alphabet_size):
    """
    Return array B, such that B[a] is
    the total number of occurrences of all characters <= a in T.
    """
    B = np.zeros(alphabet_size, dtype=np.uint64)
    for a in T:
        B[a] += 1
    for a in range(1, alphabet_size):
        B[a] += B[a-1]
    return B

@njit
def compute_types(T):
    """Return the array of position types (SMALLER=0, LARGER=1) for input T."""
    n = len(T)
    types = np.zeros(n, dtype=np.uint8)  # initialize to SMALLER, especially types[n-1]
    for i in range(n-2, -1, -1):
        types[i] = LARGER if T[i] > T[i+1] else SMALLER if T[i] < T[i+1] else types[i+1]
    return types


@njit
def find_lms_positions(types):
    n = len(types)
    # count the number of LMS positions first
    m = 0
    for p in range(1, n):
        m += (types[p] == SMALLER and types[p-1] == LARGER)
    # allocate array of just the correct size m
    lms_positions = np.empty(m, dtype=np.int64)
    # now fill the array with the actual LMS positions
    m = 0
    for p in range(1, n):
        if types[p] == SMALLER and types[p-1] == LARGER:
            lms_positions[m] = p
            m += 1
    return lms_positions


@njit
def phase2(T, B0, types, lms, pos):
    """
    Given a text 'T', cumulative bucket sizes 'B0',
    the type array 'types', array 'lms'
    of lexicographically sorted LMS positions,
    sort *all* positions lexicographically into 'pos'.
    If 'lms' contains LMS positions in arbitrary order,
    sort at least the LMS substrings correctly into pos,
    breaking ties arbitrarily.
    """
    # 0. Initialize pos by inserting LMS positions,
    B = B0.copy()  # working copy of C, to be modified
    initialize_pos_from_lms(T, B, lms, pos)
    # 1. Do a left-to-right induction scan for L-positions,
    B[:] = B0[:]  # re-set B to a clean working copy of C
    induce_L_positions(T, B, types, pos)
    # 2. Do a right-to-left induction scan for S-positions.
    B[:] = B0[:]  # re-set B to a clean working copy of C
    induce_S_positions(T, B, types, pos)
    # Result: pos has been modified as desribed.

@njit
def initialize_pos_from_lms(T, B, lms, pos):
    pos[:] = -1  # set everything to "unknown"
    # Insert the given LMS positions at the right end of their buckets
    for p in lms[::-1]:
        a = T[p]  # character determines the bucket
        B[a] -= 1
        pos[B[a]] = p

@njit
def induce_L_positions(T, B, types, pos):
    # Left-to-right scan: Induce L-positions from LMS-positions
    n = len(T)
    for r in range(n):
        p = pos[r]
        if p <= 0: continue  # unknown or 0 -> skip
        if types[p-1] == SMALLER: continue  # skip S positions
        a = T[p-1]  # determine bucket
        assert a != 0  # a cannot be the sentinel because that is an S-position
        pos[B[a-1]] = p-1
        B[a-1] += 1

@njit
def induce_S_positions(T, B, types, pos):
    # Right-to-left scan: Induce S-positions from L-positions
    n = len(T)
    for r in range(n-1, -1, -1):
        p = pos[r]
        if p == 0: continue  # skip position 0, because p-1 does not exist
        assert p != -1  # we must never encounter an undetermined entry here
        if types[p-1] == LARGER: continue  # skip L positions
        a = T[p-1]  # determine bucket
        B[a] -= 1
        pos[B[a]] = p-1


@njit
def lms_substrings_unequal(T, types, p1, p2):
    """Return True if and only if LMS substrings at p1 and p2 in T are unequal"""
    assert p1 != p2  # illegal to call with the same position
    is_lms_p1 = is_lms_p2 = False
    while True:
        if T[p1] != T[p2]:
            return True  # unequal
        if types[p1] != types[p2]:
            return True  # unequal also if types disagree, efficiency boost!
        if is_lms_p1 and is_lms_p2:
            return False  # everything was equal
        p1 +=1; p2 += 1  # look at next positions
        is_lms_p1 = (types[p1]==SMALLER) and (types[p1-1]==LARGER)
        is_lms_p2 = (types[p2]==SMALLER) and (types[p2-1]==LARGER)
        if is_lms_p1 and is_lms_p2:
            continue  # the following check is final!
        if is_lms_p1 or is_lms_p2:
            return True  # unequal; only one of the LMS strings ends


@njit
def reduce_text(T, alphabet_size, types, pos, lms_positions):
    n, m = len(pos), len(lms_positions)
    # Create array of lexicographic names to represent the correspondence
    # between LMS substrings and reduced alphabet.
    # names[p] gives the new character number
    # for the LMS substring starting at position p in the original text T.
    names = np.full(n, -1, dtype=np.int64)
    j = 0
    last_lms = n-1  # We know that there is an LMS substring (sentinel) at n-1
    names[last_lms] = 0  # It always gets the smallest possible name: 0
    reduced_alphabet_size = 1  # so far, only the sentinel 0
    # go through the suffixes lexicographically, w/o the sentinel
    for r in range(1, n):
        p = pos[r]
        # if p is not an LMS position, skip it; otherwise assign a name
        if p==0 or types[p]!=SMALLER or types[p-1]!=LARGER: continue
        lms_positions[j] = p  # build lexicographic order of lms_positions in-place
        j += 1
        # If the LMS substring at p is different from the last one, increment the name first
        if lms_substrings_unequal(T, types, last_lms, p):
            reduced_alphabet_size += 1
        names[p] = reduced_alphabet_size - 1
        last_lms = p
    # construct reduced text
    R = np.zeros(m, dtype=np.uint32)
    position_map = np.zeros(m, dtype=np.int64)
    i = 0
    for p, name in enumerate(names):
        if name != -1:
            R[i] = name
            position_map[i] = p
            i += 1
    return R, reduced_alphabet_size, position_map
    # In addition, lms_positions has been sorted lexicographically!


@njit
def phase1(T, B, types, lms_positions, pos):
    # Run phase2 (induced sorting) using the unordered(!) LMS positions.
    # This will not necessarily result in a correct suffix array,
    # but at least sort the LMS subSTRINGS correctly;
    # equal LMS substrings may appear in random order.
    alphabet_size = len(B)
    phase2(T, B, types, lms_positions, pos)
    # Compute the reduced text from the LMS substrings ("lexicographic naming")
    (R, reduced_alphabet_size, position_map) \
        = reduce_text(T, alphabet_size, types, pos, lms_positions)
    # check if there are equal LMS substring; if yes, then recurse on reduced text R.
    if len(R) != reduced_alphabet_size:
        reduced_pos = sais_main(R, reduced_alphabet_size)
        # Re-map reduced_pos to original text positions;
        # these are the lms_positions in lexicographic order,
        # so we overwrite lms_positions with the correct sorted order.
        for i, redp in enumerate(reduced_pos):
            lms_positions[i] = position_map[redp]


@njit
def sais_main(T, alphabet_size):
    """
    Compute and return suffix array pos of a text T.
    T must be a numeric array and end with a unique sentinel 0.
    It's real alphabet must be {1, 2, ..., alphabet_size-1}.
    This is ensured by not calling this function directly,
    but calling compute_pos_sais(T).
    """
    # Allocate suffix array (initialized later)
    print("sais_main:", len(T), alphabet_size)
    pos = np.empty(len(T), dtype=np.int64)
    # B[a]: total number of characters in T that are <= a
    B = count_cumulative_characters(T, alphabet_size)
    # Compute types and LMS positions (ascending)
    types = compute_types(T)
    lms_positions = find_lms_positions(types)
    # Run phase 1; this will sort lms_positions lexicographically in-place.
    # Note that phase1() may recurse into sais_main() with a reduced text.
    print("phase 1")
    phase1(T, B, types, lms_positions, pos)
    # Run phase2 (induced sorting) using the correctly sorted LMS positions.
    # This will result in the correct suffix array pos.
    print("phase 2")
    phase2(T, B, types, lms_positions, pos)
    return pos


def compute_pos_sais(T):
    # Given a string: do smart translation to bytes, check sentinel.
    # Otherwise, just assume that a valid array has been given.
    if len(T) == 0:
        raise ValueError("ERROR: empty text, must at least contain a sentinel!")
    if isinstance(T, str):
        t = T.encode("ASCII")  # fails if non-ASCII present!
        occ = Counter(t)
        alphabet = sorted(occ)  # sorted characters
        if t[-1] != alphabet[0]:
            raise ValueError(f"ERROR: last character '{t[-1]}' is not the smallest in alphabet {alphabet}")
        if occ[t[-1]] != 1:
            raise ValueError(f"ERROR: last character '{t[-1]}' in T is not unique!")
        translator = bytes.maketrans(bytes(alphabet), bytes(range(len(alphabet))))
        T = np.frombuffer(t.translate(translator), dtype=np.uint8)
    # from here on, T is a numeric array or buffer
    if T[-1] != 0:
        raise ValueError(f"ERROR: last text element '{T[-1]}' must be zero!")
    if np.sum(T == 0) > 1:
        raise ValueError(f"ERROR: sentinel 0 is not unique in T")
    alphabet_size = max(T) + 1
    return sais_main(T, alphabet_size)


def test_text(T):
    print(T, compute_pos_sais(T), "-"*len(T), sep="\n")


def test_random(length, asize):
    from numpy.random import default_rng
    rg = default_rng()
    T = rg.integers(low=1, high=asize+1, size=length)
    T[-1] = 0  # sentinel
    pos = compute_pos_sais(T)
    print("checking...")
    assert np.all(np.sort(pos) == np.arange(length))


if __name__ == "__main__":
    test_text("miississippii$")
    test_text("ACATACATACATACCATACATACAT$")
    #test_text("gccttaacattattacgccta$")
    # test_random(300_000_000, 4)