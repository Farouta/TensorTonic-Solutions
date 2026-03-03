import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 4
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.word_to_id["<PAD>"]=0
        self.id_to_word[0]="<PAD>"
        self.unk_token = "<UNK>"
        self.word_to_id["<UNK>"]=1
        self.id_to_word[1]="<UNK>"
        self.bos_token = "<BOS>"
        self.word_to_id["<BOS>"]=2
        self.id_to_word[2]="<BOS>"
        self.eos_token = "<EOS>"
        self.word_to_id["<EOS>"]=3
        self.id_to_word[3]="<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        unique_words=[]
        for text in texts :
            words=text.split()
            for word in words:
                unique_words.append(word.lower())
        for word in unique_words:
            if word not in self.word_to_id :
               self.word_to_id[word]=self.vocab_size
               self.id_to_word[self.vocab_size]=word
               self.vocab_size +=1
        pass

    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """

        out=[]
        words=text.split()
        for word in words :
            word=word.lower()
            if word in self.word_to_id :
                out.append(self.word_to_id[word])
            else :
                out.append(self.word_to_id["<UNK>"])


        return out
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        out=[]
        sentance=""
        for id in ids :
            out.append(self.id_to_word[id])
        sentance=" ".join(out)
        return sentance