from collections import defaultdict
import json

class Tokenizer:

    def __init__(self, corpus:str, nb_merge:int=5, verbose:bool=False):
        self.corpus = corpus
        self.nb_merge = nb_merge
        self.verbose = verbose
        self.merge = []

    def encode(self):
        """
            Implement the BPE algorithm.
        """
        # Slice every words in the corpus
        words = self.corpus.split()
        if self.verbose:
            print(f"Words : {words}")
        # transforme every word in a list of character with a tag </w> to precise the end of the word
        words = [[c for c in word] + ["</w>"] for word in words]
        if self.verbose:
            print(f"Words : {words}")
        # Create a dict for the vocab and there frequency
        vocab_freq = {}
        for word in words:
            word_tuple = tuple(word)
            if word_tuple not in vocab_freq:
                vocab_freq[word_tuple] = 1
            else:
                vocab_freq[word_tuple] += 1

        if self.verbose:
            print(f"vocab_freq : {vocab_freq}")
        # Count every pair frequency
        def get_pair_frequencies(vocab:dict):
            pair_freq = {}
            for word, freq in vocab.items():
                pairs = {(word[i], word[i+1]) for i in range(len(word)-1)}
                for pair in pairs:
                    if pair in pair_freq:
                        pair_freq[pair] += 1
                    else:
                        pair_freq[pair] = 1
            
            if self.verbose:
                print(f"pair_freq : {pair_freq}")
            return pair_freq
        
        # merge in the vocab all occurence of the max pair 
        def merge_pair(vocab:dict, pair:tuple):
            new_vocab_freq = defaultdict(int)

            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                        new_word.append(word[i] + word[i+1])
                        i+=2
                    else:
                        new_word.append(word[i])
                        i+=1
                new_vocab_freq[tuple(new_word)] += freq 

            if self.verbose:
                print(f"vocab freq : {vocab_freq}")
            return dict(new_vocab_freq)        
        

        for i in range(self.nb_merge):
            pair_freq = get_pair_frequencies(vocab_freq)
            if not pair_freq:
                break
            
            pair_max = max(pair_freq, key=pair_freq.get)
            if self.verbose:
                print(f"pair Max : {pair_max}")

            self.merge.append(pair_max)
            vocab_freq = merge_pair(vocab_freq,pair_max)
        

        vocab_as_str = {"".join(token): freq for token, freq in vocab_freq.items()}

        tokenizer_data = {
            "vocab": vocab_as_str,
            "merges": self.merge
        }

        with open("tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        


    def decode(self):
        ...