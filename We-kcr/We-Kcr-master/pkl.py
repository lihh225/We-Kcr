import pickle
import numpy as np
with open('./data/residue2idx2.pkl', 'rb')as f:
    a=pickle.load(f)
print(a)
#{'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, 'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9, 'G': 10, 'K': 11, 'Y': 12, 'P': 13, 'H': 14, 'Z': 15, 'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20, 'R': 21, 'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27}
# dic={'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3, 'B': 4, 'Q': 5, 'I': 6, 'D': 7, 'M': 8, 'V': 9, 'G': 10, 'K': 11, 'Y': 12, 'P': 13, 'H': 14, 'Z': 15, 'W': 16, 'U': 17, 'A': 18, 'N': 19, 'F': 20, 'R': 21, 'S': 22, 'C': 23, 'E': 24, 'L': 25, 'T': 26, 'X': 27,'O':28}
# with open('./data/residue2idx2.pkl', 'wb')as f:
#     pickle.dump(dic,f)