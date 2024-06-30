from src.utils import *
import torch
from src.model.decoder.triplet_scorer.TransEScorer import TransEScorer
from src.model.decoder.triplet_scorer.RotatEScorer import RotatEScorer
from src.model.decoder.triplet_scorer.ComplExScorer import ComplExScorer
from src.model.decoder.triplet_scorer.DistMultScorer import DistMultScorer

class BaseDecoder(torch.nn.Module):
    def __init__(self, args, kg):
        super().__init__()
        self.args = args
        self.kg = kg

    def forward(self, feat, mode=None):
        ''''''
        return None
