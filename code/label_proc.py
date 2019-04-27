import numpy as np
import string


# calculate the statistics of a label set
def label_statistics(letters, label):
    char_dict = {}
    total_word_len = 0
    total_sentence_len = 0
    word_cnt = 0
    sentence_cnt = 0

    for letter in letters:
        char_dict[letter] = 0

    for sentence in label:
        length = len(sentence)
        char_dict[' '] += length - 1

        char_dict['{'] += 1
        char_dict['}'] += 1

        total_sentence_len += length
        sentence_cnt += 1
        for word in sentence:
            total_word_len += len(word)
            word_cnt += 1
            word = "".join(map(chr, word))
            for char in word:
                if char in char_dict.keys():
                    char_dict[char] += 1
                else:
                    print("char {} not found".format(char))

    avg_word_len = total_word_len / word_cnt
    avg_sentence_len = total_sentence_len / sentence_cnt

    print("Avg word len is {}, Avg sentence len is {}".format(round(avg_word_len, 2), round(avg_sentence_len, 2)))

    for key in char_dict.keys():
        print(key, round(char_dict[key] / len(label), 2))

    return


# process the labels into lists of ints
def process_label(char_idx, label):
    label_embed = []

    for sentence in label:
        st_embed = []
        st_embed.append(char_idx['{'])
        first = True
        for word in sentence:
            if not first:
                st_embed.append(char_idx[' '])
            else:
                first = False
            word = "".join(map(chr, word))
            for letter in word:
                st_embed.append(char_idx[letter])

        st_embed.append(char_idx['}'])
        st_embed = np.array(st_embed)

        label_embed.append(st_embed)
    return label_embed


# take an array and convert it to sentence
def toSentence(arr):
    idx_char = np.load(idx_char_path, allow_pickle=True)
    idx_char = idx_char.item()
    sentence = []
    for e in arr:
        sentence.append(idx_char[int(e)])

    sentence = ''.join(sentence)
    return sentence


train_label_path = '../data/train_transcripts.npy'
val_label_path = '../data/dev_transcripts.npy'
idx_char_path = '../data/idx_char.npy'
char_idx_path = '../data/char_idx.npy'


if __name__ == '__main__':
    letters = np.array([_ for _ in string.ascii_uppercase[:26]])
    letters = np.append(letters, ' ')
    letters = np.append(letters, '.')
    letters = np.append(letters, '-')
    letters = np.append(letters, '_')
    letters = np.append(letters, '\'')
    letters = np.append(letters, '+')
    letters = np.append(letters, '{')  # start of sentence(id=32)
    letters = np.append(letters, '}')  # end of sentence(id=33)

    print("There are {} letters: {}".format(len(letters), letters))

    # label_statistics(letters, train_label)
    # label_statistics(letters, val_label)

    # get a map of encoding idx and chars
    idx_char = {}
    char_idx = {}
    for i in range(len(letters)):
        idx_char[i] = letters[i]
        char_idx[letters[i]] = i

    np.save(idx_char_path, idx_char)
    np.save(char_idx_path, char_idx)
    print("idx_char:", idx_char)
    print("char_idx:", char_idx)

    # process the labels
    train_label = np.load(train_label_path, allow_pickle=True, encoding='bytes')
    val_label = np.load(val_label_path, allow_pickle=True, encoding='bytes')

    train_label_embeds = process_label(char_idx, train_label)
    val_label_embeds = process_label(char_idx, val_label)

    np.save('../data/train_label_emb.npy', train_label_embeds)
    print("done with train labels")
    np.save('../data/val_label_emb.npy', val_label_embeds)
    print("done with val labels")
