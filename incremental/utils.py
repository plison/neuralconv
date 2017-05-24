import numpy as np, pickle
from nltk.stem.snowball import SnowballStemmer

"""
Utility functions for extracting features from the TAKE dataset.
"""

ALL_FEATURES =  ['col', 'color', 'grid', 'row', 'type', 'x', 'y']

COLOR_CODES = {'blue': 0, 'cyan': 1, 'gray': 2, 'green': 3, 
               'magenta': 4, 'red': 5, 'yellow': 6}
TYPE_CODES = {'F': 0, 'I': 1,  'L': 2, 'N': 3,  'P': 4, 'T': 5,
              'U': 6,'V': 7, 'X': 8, 'Y': 9, 'Z': 10}

try:
    dico = pickle.load(open("dico.pkl", "rb"))
except:
    dico = {}
stemmer =  SnowballStemmer("german")

def encode_tokens(utterance, length, remove_sil=True):  
    """Transforms a list of tokens into a sequence of integers, where
    each integer is an identifier for the word stem. The list is then padded
    on the left to create an array of fixed length"""
    tokens = []
    for w in utterance:
        if remove_sil and w=="<sil>":
            continue
        w = stemmer.stem(w)
        if w not in dico:
            dico[w] = len(dico) + 1
        
        tokens.append(dico[w])
    if length is None:
        return tokens
    tokens = tokens[-length:] if len(tokens) > length else tokens
    padded_tokens = np.zeros(length)
    padded_tokens[-len(tokens):] = tokens
    return padded_tokens


def encode_token(token, remove_sil=True):
    """Transform the token into its unique identifier"""
    
    if remove_sil and token=="<sil>":
        return 0
    token = stemmer.stem(token)
    if token not in dico:
        print("Token not in vocabulary list:", token)
        return 0
    else:
        return dico[token]


def encode_bag_of_words(utterance, nb_tokens=306):
    """Transform the utterance into a sparse list of word counts"""
    
    bag_of_words = np.zeros(nb_tokens)
    for token in utterance:
        encoded_token = encode_token(token)
        if encoded_token  > 0:
            bag_of_words[int(encoded_token)-1] += 1
    return bag_of_words


def encode_features(obj):
    """Transform the object description into a fixed sequence of feature
    values, using one-hot encoding for the color and type."""
    
    one_hot_list = []
    
    one_hot_col = [0]*3
    one_hot_col[obj["col"]] = 1
    one_hot_list += one_hot_col
    
    one_hot_color = [0]*len(COLOR_CODES)
    one_hot_color[COLOR_CODES[obj["color"]]] = 1
    one_hot_list += one_hot_color
    
    if obj["grid"] == "grid1":
        one_hot_list += [1, 0, 1, 0]
    elif obj["grid"] == "grid2":
        one_hot_list += [1, 0, 0, 1]
    elif obj["grid"] == "grid3":
        one_hot_list += [0, 1, 1, 0]
    elif obj["grid"] == "grid4":
        one_hot_list += [0, 1, 0, 1]
        
    one_hot_row = [0]*3
    one_hot_row[obj["row"]] = 1
    one_hot_list += one_hot_row
    
    one_hot_type = [0]*len(TYPE_CODES)
    one_hot_type[TYPE_CODES[obj["type"]]] = 1
    one_hot_list += one_hot_type
    
    one_hot_list += [(obj["x"]-175.0)/1715.0]
    one_hot_list += [(obj["y"]-125.0)/915.0]
    
    return np.array(one_hot_list)

