import torch
from .utils import *
from .model.model_process import CrossLinkPredictionTestBatchProcessor, RPG_filler


class Tester():
    def __init__(self, args, kg, model):
        self.args = args
        self.kg = kg
        self.model = model
        self.test_processor = CrossLinkPredictionTestBatchProcessor(args, kg)

    def test(self):
        torch.cuda.empty_cache()
        self.args.valid = False
        res = self.test_processor.process_epoch(self.model)
        self.args.valid = True

        return res


class RPGFiller():
    def __init__(self, args, kg, model):
        self.args = args
        self.kg = kg
        self.model = model
        self.boot_processor = RPG_filler(args, kg)

    def fill_cross_KG_part(self):
        return self.boot_processor.process_epoch(self.model)
