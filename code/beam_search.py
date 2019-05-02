import numpy as np
from queue import PriorityQueue as PQ


# append the input char to the list
def append_char(char_list, char):
    if not char_list is None:
        char_list = np.append(char_list, char)
    else:
        char_list = [char]
    return char_list


# calculate all the sub nodes for nodes in the searcher
# the parents will disappear, leaving only their children
# put any eos nodes into pool
def expand_searcher(searcher, model):
    print("into expand!")
    global pool
    new_searcher = PQ()
    while searcher.qsize() > 0:
        parent = searcher.get()
        parent_prob = parent[0]
        parent_path = parent[1]

        child_probs = model(parent_prob)
        for idx in range(len(child_probs)):
            child_prob = child_probs[idx] * parent_prob
            child_path = append_char(parent_path, idx)

            if idx == end_symb:
                pool.put((child_prob, child_path))

            else:
                new_searcher.put((child_prob, child_path))

    return new_searcher


# leave only top beam width nodes in the searcher
def clear_searcher(searcher, model):
    print("into clear!")
    global beam_width
    # if no more than beam width, return the input searcher
    if searcher.qsize() <= beam_width:
        print("no clear, the searcher has only {} nodes").format(searcher.qsize())
        return searcher

    # leave only top beam width nodes
    new_searcher = PQ()

    for i in range(beam_width):
        node = searcher.get()
        new_searcher.put(node)

    return searcher


if __name__ == '__main__':
    beam_width = 4
    searcher = PQ()
    pool = PQ()

    char = 32
    end_symb = 33

    first_node = (-1, append_char(None, char))
    print("first node is:", first_node)
    searcher.put(first_node)

    while pool.qsize() < beam_width:
        searcher = expand_searcher(searcher)
        searcher = clear_searcher(searcher)

    while pool.qsize() > 0:
        node = pool.get()
        print(node)

