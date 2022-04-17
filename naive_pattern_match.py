import argparse
import numpy as np


def naive_pattern_matching(P, T):
  n = comps = i = 0
  positions = []
  found = False
  while i <= len(T) - len(P):
    for j in range(len(P)):
      comps+=1
      if T[j+i] != P[j]:
        i+=1
        break
      elif T[j+i] == P[j] and j == len(P)-1:
          found = True
          n+=1
          positions.append(i) #append the position
          i+=1
          break
  return found, n, positions, comps 


def get_text(text=None,textfile=None):
  if textfile is not None:
    with open(textfile) as file:
      text = file.read()
  #remove the spaces
  if " " in text:
    text = text.replace(" ","")
  #remove the "\n"
  if "\n" in text:
    text = text.replace("\n","")
  return text


def get_patterns(pattern=None,patternfile=None):
  if patternfile is not None:
    with open(patternfile) as file:
      #read each line within the file
      Ps = file.readlines()
      #Remove empty lines & "\n" from the lines
      Ps = list(map(lambda x:x.strip(),Ps))
  else:
    Ps = []
    #remove the spaces
    if " " in pattern:
      pattern = pattern.replace(" ","")
    Ps.append(pattern)
  return Ps


def main(args):
    T = get_text(args)
    Ps = get_patterns(args)

    for P in Ps:  # iterate over patterns
        if len(P) == 0: continue  # skip empty patterns
        print(f"> {P}")
        found, n , positions, comps = naive_pattern_matching(P, T)
        print(f"Pattern {P}:\n Found: {found}\n Occurred: {n} times \n Positions: {positions}\n Comparisons: {comps}")


def get_argument_parser():
    p = argparse.ArgumentParser(description="DNA naive pattern matching")
    pat = p.add_mutually_exclusive_group(required=True)
    pat.add_argument("-P", "--pattern",
        help="immediate pattern to be matched")
    pat.add_argument("-p", "--patternfile",
        help="name of file containing patterns (one per line)")
    txt = p.add_mutually_exclusive_group(required=True)
    txt.add_argument("-T", "--text",
        help="immerdiate text to be searched")
    txt.add_argument("-t", "--textfile",
        help="name of file containing text")
    return p


if __name__ == "__main__":
    main(get_argument_parser().parse_args())
