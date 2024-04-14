#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import torch

from data.othello import get_ood_game
from . import const

class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self):
        data = const.DATA_DIR / 'data.pt'
        if not data.exists(): self.generate()
        self.data = torch.load(data)

        self.central = [27, 28, 35, 36]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx, :-1], self.one_hot(self.data[idx, -1].item())

    def generate(self):
        games = [game[-1].text.upper() for game in ET.parse(const.DATA_DIR / 'woc2006.xml').getroot()[-1]]
        torch.save(torch.tensor([[*x, 1] for x in map(self.to_index, games) if x is not None] + [[*get_ood_game(0), 0] for _ in range(len(games))]), const.DATA_DIR / 'data.pt')

    def as_seqs_str(self):
        return self.data[:, :-1]

    def as_seqs_int(self, data=None):
        if data is None: data = self.data[:, :-1].detach().clone()
        for central in sorted(self.central, reverse=True):
            data[data > central] -= 1
        return data + 1 # pass is 0, offset by 1

    @staticmethod
    def one_hot(idx):
        y = torch.zeros(2)
        y[idx] = 1.

        return y

    @staticmethod
    def to_index(game):
        res = []
        for alph, num in zip(game[::2], game[1::2]):
            res.append((ord(alph) - 65)*8 + int(num) - 1)
        if len(res) != 60:
            # res.extend([-1,]*(60-len(res)))  # bad practice, passed moves need to injected not appended
            return None  # skip games with 'pass' moves
        return res


if __name__ == '__main__':
    from IPython import embed

    ds = ContrastiveDataset()
    embed()
