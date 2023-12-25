import random
import pandas as pd
from transformers import BertTokenizer


class KaraMaker:
    def __init__(self):
        self.vocab = self.get_vocabs()
        self.df = self.make_dataset()

    def get_vocabs(self) -> list:
        new_list = []
        vocab = BertTokenizer.from_pretrained("bert-base-cased").vocab
        for key, value in vocab.items():
            if value < 1103:
                continue
            else:
                new_list.append(key)
        return new_list

    def make_sentence(self):
        sentence_len = random.randint(5, 16)
        sampled = random.choices(self.vocab, k=sentence_len)
        sentence = " "
        sentence = sentence.join(sampled)
        sentence = sentence + "."
        sentence = sentence.replace("#", "")
        return sentence

    def make_paragraph(self):
        paragraph_len = random.randint(5, 16)
        paragraph_list = []
        for _ in range(paragraph_len):
            paragraph_list.append(self.make_sentence())

        paragraph = " "
        paragraph = paragraph.join(paragraph_list)
        return paragraph

    def make_dataset(self, length=10000):
        new_dict = {"product": {}, "description": {}}
        for idx in range(length):
            new_dict["product"][idx] = str(idx)
            new_dict["description"][idx] = self.make_paragraph()
        df = pd.DataFrame(new_dict)
        return df


def main():
    maker = KaraMaker()
    maker.df.to_csv("./data/product_description.csv")

    pass


if __name__ == "__main__":
    main()
