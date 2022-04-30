from assignment03 import ACNode, AC_build, search_with_AC

def test_delta():
    P = ["AAB", "ABABBAB", "BAA"]
    root = AC_build(P)
    assert root.delta("A").delta("B").depth == 2
    assert root.delta("A").delta("C").lps is None

def test_search_with_AC():
    P = ["AAB", "ABABBAB", "BAA"]
    T = "AABABABBABABBABBBBBABABBAA"
    assert list(search_with_AC(P,T)) == [(0, 3, 0), (3, 10, 1), (8, 15, 1), (23, 26, 2)]

    P = ["it", "toy", "bit", "you", "unit", "o"]
    T = "o joy, a toy, to you, it was a bit of a unit"
    assert list(search_with_AC(P,T)) == [(0, 1, 5), (3, 4, 5), (10, 11, 5), (9, 12, 1), (15, 16, 5), (18, 19, 5), (17, 20, 3), (22, 24, 0), (31, 34, 2), (32, 34, 0), (35, 36, 5), (40, 44, 4), (42, 44, 0)]
