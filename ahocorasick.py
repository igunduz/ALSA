# assumes P is a list of strings

import collections   # has a deque for breadth-first-search
import argparse

#####################################################
# Aho-Corasick

class ACNode():
    
    def __init__(self, parent=None, letter=None, depth=0, label=""):
        self.targets = dict()  # children of this node
        self.lps = None        # lps link of this node
        self.parent = parent   # parent of this node
        self.letter = letter   # letter between parent and this node
        self.out = []          # output function of this node
        if parent is None:
            self.depth = depth # number of chars from root to this n ode
            self.label = label # string from root to this node
        else:
            self.depth = parent.depth + 1
            self.label = parent.label + letter

        
    def delta(self, a):
        """
        Transit to next node upon processing character a.
        INPUT:
        - self is the object. you can acces parameters like parent using self.parent
        - a: text character
        OUTPUT:
        - q: the next state (another ACNode instance)
        TODO: Implement the delta function for the current node:
        What is the target of self after reading character a?
        """
        q = self #paste selft to q
        while a not in q.targets and q.lps is not None:
            q = q.lps 
        if a in q.targets: 
            q = q.targets[a]
        return q

    def bfs(self):
        """yields each node below and including self in BFS order"""
        Q = collections.deque([self])
        while len(Q) > 0:
            node = Q.popleft()
            yield node
            Q.extend(node.targets.values())

    def __str__(self):
        s = ""
        for (i,node) in enumerate(self.bfs()):
            lps = node.lps.label if node.lps!=None else "[none]"
            s += "{0}: label={1}, targets={2}, lps={3}, output={4}\n".format(
                i, node.label, list(node.targets.keys()), lps, node.out)
        return s


def AC_build(P):
    """build AC autmaton for list of patterns P, return its root node."""
    # Build a root for the trie
    root = ACNode()
    # Build the trie, pattern by pattern
    for (i,p) in enumerate(P):
        node = root
        for a in p:
            if a in node.targets:
                node = node.targets[a]
            else:
                newnode = ACNode(parent=node, letter=a)
                node.targets[a] = newnode
                node = newnode
        node.out.append(i)
    # Walk through trie in BFS-order to build lps
    for node in root.bfs():
        if node.parent is None: continue
        node.lps = node.parent.lps.delta(node.letter) if node.depth>1 else root
        node.out.extend(node.lps.out)
    return root

    
            
def search_with_AC(P, T):
    """
    INPUT:
    - List of Patterns P
    - Text T
    OUTPUT:
    - yield each triple (start, stop, pattern_index)
    """
    root = AC_build(P)
    for i in range(len(T)):
        c = T[i]
        root = root.delta(c)
        for out in root.out:
            yield(i + 1 - len(P[out]),i + 1 , out)


def main(args):
    T = args.text
    P = args.pattern
    ret = search_with_AC(P,T)
    print(list(ret))

def get_argument_parser():
    p = argparse.ArgumentParser(description="DNA Motif Searcher")
    p.add_argument("-P", "--pattern", required=True, nargs="+",
        help="immediate pattern to be matched")
    p.add_argument("-T", "--text", required=True,
        help="immerdiate text to be searched")
    return p

if __name__ == "__main__":
    main(get_argument_parser().parse_args())
