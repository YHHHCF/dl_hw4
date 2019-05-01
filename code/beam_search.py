import numpy as np
from queue import PriorityQueue as PQ


# get prob from input char
def model(char):
    prob = np.random.rand(34)
    prob = prob / sum(prob)
    return prob


# append the input char to the list
def append_char(char_list, char):
    if char_list:
        char_list = np.append(char_list, char)
    else:
        char_list = [char]
    return char_list


def expand_searcher(searcher):
    return searcher


def clear_searcher(searcher):
    return searcher


if __name__ == '__main__':
    beam_width = 32
    searcher = PQ()
    pool = PQ()

    char = 32
    end_symb = 33

    while pool.qsize() < beam_width:
        searcher.put((1, append_char(None, char)))
        prob = model(char)
        searcher = expand_searcher(searcher)
        searcher = clear_searcher(searcher)
