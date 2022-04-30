"""Shift-And and Shift-Or pattern matching"""

import argparse  # for command line interface

def build_nfa_and(P):
    """Build an NFA from a pattern"""
    """
    Input:
        Pattern P (String)
    Output:
        mask: Dictionary of all masks for all different characters in the pattern (dict)
    """
    
    unique = set(P) #masks all unique characters in the text
    masks = dict()
    state_sets = range(0,len(P))
    for element in unique:
        masks[element] = ""
    P = P[::-1] #reverse the text
    bit = 2
    for element in masks:
        bit = bit * 2
        for state in state_sets:
            if element is P[state]:
                masks[element] = str(masks[element]) + "1"
            else:
                masks[element] = str(masks[element]) + "0"
        masks[element] = int(masks[element], 2)
    return masks,bit



def build_nfa_or(P):
    """
    Input:
        Pattern P (String)
    Output:
        mask: Dictionary of all masks for all different characters in the pattern (dict)
        accept:
    """

    unique = set(P) #store all unique characters in the text
    masks = dict()
    state_sets = range(0,len(P))
    for element in unique:
        masks[element] = ""
    bit = 2
    P = P[::-1] #reverse the text
    for element in masks:
        bit = bit * 2
        for state in state_sets:
            if element is P[state]:
                masks[element] = str(masks[element]) + "0"
            else:
                masks[element] = str(masks[element]) + "1"
        masks[element] = int(masks[element], 2)
    return masks,bit


def shift_and(mask, accept, text, N):
    """
    Input:
        masks: Dictionary of all masks (dict) and their accepting state
        text: Text (String)
        N: Maximum number of positions to store in results
    Output:
        results: List of all positions at which a pattern ends (list)
        k: number of matches (int)
    """
    k = 0
    results = []
    accept = max(masks.values())
    D = 0 # bit-mask of active states
    for i, c in enumerate(text):
        D = ((D << 1) | 1) & masks[c]
        if (D & accept) !=0:
            results.append(i)
            k+=1 #increase the match
    if N > 1:
        results = results[1:N+1] #redifine results
        k = k - 1  #delete the extra match
    return k, results


def shift_or(mask, accept, text, N):
    """
    Input:
        masks: Dictionary of all masks (dict) and their accepting state (int)
        text: Text (String)
        N: Maximum number of positions to store in results
    Output:
        results: List of all positions at which a pattern ends (list)
        k: number of matches (int)
    """
    k = 0
    results = []
    D = 0 # bit-mask of active states
    for i, c in enumerate(text):
        D = ((D << 1) | (masks[c]))
        if (D & accept) != 0 and i != 1:
            results.append(i)
            k = k + 1 #increase the match
    return k, results

    
def get_text(args):
    if args.text is not None:
        return args.text
    with open(args.textfile, "r") as ftext:
        text = ftext.read()
    return text


def get_patterns(args):
    if args.pattern is not None:
        return [args.pattern]  # list with single item
    with open(args.patternfile, "r") as fpat:
        Ps = [pattern.strip() for pattern in fpat.readlines()]
    return Ps


def main(args):
    alg = args.algorithm
    T = get_text(args)  # bytes object
    Ps = get_patterns(args)  # list of bytes objects
    build_nfa = build_nfa_and if alg == "and" else build_nfa_or
    find_matches = shift_and if alg == "and" else shift_or
    NRESULTS = args.maxresults
    for P in Ps:  # iterate over patterns
        if len(P) == 0: continue  # skip empty patterns
        nfa = build_nfa(P)
        nresults, results = find_matches(*nfa, T, NRESULTS)
        if nresults > NRESULTS:
            print("! Too many results, showing first {NRESULTS}")
            nresults = NRESULTS
        print(*list(results[:nresults]), sep="\n")


def get_argument_parser():
    p = argparse.ArgumentParser(description="DNA Motif Searcher")
    pat = p.add_mutually_exclusive_group(required=True)
    pat.add_argument("-P", "--pattern",
        help="immediate pattern to be matched")
    pat.add_argument("-p", "--patternfile",
        help="name of file containing patterns (one per line)")
    txt = p.add_mutually_exclusive_group(required=True)
    txt.add_argument("-T", "--text",
        help="immerdiate text to be searched")
    txt.add_argument("-t", "--textfile",
        help="name of file containing text (will be read in one piece)")
    p.add_argument("-a", "--algorithm", metavar="ALGORITHM",
        default="and", choices=("and", "or"),
        help="algorithm to use ('and' (default), 'or')")
    p.add_argument("--maxresults", "-R", type=int, default=10_000,
        help="maximum number of results to show (10_000)")
    return p


if __name__ == "__main__":
    main(get_argument_parser().parse_args())
