class Tokenizer:
    def __init__(self) -> None:
        self.char_string = (
            "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        )
        self.chars = sorted(list(set(self.char_string)))
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        stoi = {ch: i for i, ch in enumerate(self.chars)}
        itos = {i: ch for i, ch in enumerate(self.chars)}

        self.encode = lambda string: [stoi[c] for c in string]
        self.decode = lambda tokens: "".join([itos[i] for i in tokens])
