import argparse
from src.train import *
from src.test import *
import sys
torch.autograd.set_detect_anomaly(True)
import shutil
from datetime import datetime
from src.data_load.KnowledgeGraph import *
from src.model.controller import Controller


class experiment:
    def __init__(self, args):
        self.args = args

        '''1. prepare data file path, model saving path and log path'''
        self.prepare()

        '''2. load data'''
        self.kg = KnowledgeGraph(args)

        '''3. create model and optimizer'''
        self.model, self.optimizer = self._create_model()
        self.start_epoch = 0

        if self.args.load_checkpoint is not None:
            self.start_epoch = self.load_checkpoint(os.path.join(self.args.load_checkpoint, 'model_best.tar'))
            self.model.args = self.args
            self.model.kg = self.kg
        self.args.logger.info(self.args)

    def _create_model(self):
        '''
        Initialize KG embedding model and optimizer.
        return: model, optimizer
        '''
        model = Controller(self.args, self.kg)
        model.to(self.args.device)
        init_param(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.l2)
        return model, optimizer

    def train(self):
        '''
        Training process
        :return: training time
        '''
        start_time = time.time()
        self.best_valid = 0.0
        self.stop_epoch = 0
        trainer = Trainer(self.args, self.kg, self.model, self.optimizer)
        filler = RPGFiller(self.args, self.kg, self.model)

        print("Start Training ===============================>")
        '''Training iteration'''
        for epoch in range(self.start_epoch, int(self.args.epoch_num)):
            if self.args.RPG and epoch >= self.args.warmup and (epoch-self.args.warmup) % self.args.RPG_update_span==0:
                same, inverse = filler.fill_cross_KG_part()
                if self.args.use_augment:
                    trainer.train_processor.add_facts_using_relations(same, inverse)

                if epoch == self.args.warmup:
                    self.best_valid = 0
            self.args.epoch = epoch
            '''training'''
            loss, valid_res = trainer.run_epoch()
            '''early stop'''
            if self.best_valid <= valid_res[self.args.valid_metrics]:
                self.best_valid = valid_res[self.args.valid_metrics]
                self.stop_epoch = max(0, self.stop_epoch-5)
                self.save_model(is_best=True)
            else:
                self.stop_epoch += 1
                if self.stop_epoch >= self.args.patience:
                    self.args.logger.info('Early Stopping! Epoch: {} Best Results: {}'.format(epoch, round(self.best_valid*100, 3)))
                    break
            '''logging'''
            if epoch % 1 == 0:
                self.args.logger.info('Epoch:{}\tLoss:{}\tH@1:{}\tH@3:{}\tH@5:{}\tH@10:{}\tMRR:{}\tBest:{}'.format(epoch,round(loss, 3), round(valid_res['hits1'] * 100, 2), round(valid_res['hits3'] * 100, 2), round(valid_res['hits5'] * 100, 2), round(valid_res['hits10'] * 100, 2), round(valid_res['mrr'] * 100, 2), round(self.best_valid * 100,2)))
        end_time = time.time()
        training_time = end_time - start_time
        return training_time

    def test(self, load_best=True):
        self.kg.load_test()
        if load_best and self.args.load_checkpoint is None:
            best_checkpoint = os.path.join(self.args.save_path, 'model_best.tar')
            self.load_checkpoint(best_checkpoint)
        tester = Tester(self.args, self.kg, self.model)

        res = tester.test()
        print(res)
        return res

    def prepare(self):
        '''
        set the log path, the model saving path and device
        :return: None
        '''
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        args.lambda_1 = float(args.lambda_1)
        args.lambda_2 = float(args.lambda_2)
        args.alpha = float(args.alpha)

        '''set data path'''
        self.args.data_path = args.data_path + args.dataset + '/'
        self.args.save_path = args.save_path + args.dataset + '-' + args.scorer + '-' + args.encoder +'-'+ str(args.emb_dim)+'-' + str(args.margin)

        '''add logging implement to model path for ablation_study'''
        if self.args.ea_expand_training:
            self.args.save_path = self.args.save_path + '-ea_expand_training'
        if self.args.RPG:
            self.args.save_path = self.args.save_path + '-RPG'
            if not self.args.use_attn:
                self.args.save_path=self.args.save_path + '-wo attn'
            if not self.args.use_RPG_triple:
                self.args.save_path=self.args.save_path + '-wo triple'
            if not self.args.use_augment:
                self.args.save_path=self.args.save_path + '-wo augment'


        self.args.save_path = self.args.save_path + '-' + str(self.args.ea_rate) + '--' + str(self.args.learning_rate)+ '-' + str(args.seed) + '-neg_ratio-' + str(args.neg_ratio)
        # TODO ablation study

        if self.args.note != '':
            self.args.save_path = self.args.save_path + self.args.note

        if os.path.exists(args.save_path) and args.load_checkpoint is None:
            shutil.rmtree(args.save_path, True)
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        self.args.log_path = args.log_path + datetime.now().strftime('%Y%m%d/')
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)
        self.args.log_path = args.log_path + args.dataset + '-' + args.scorer + '-' + args.encoder +'-'+ str(args.emb_dim)+ '-' + str(args.margin)

        # '''add logging implement to log path for ablation_study'''
        if self.args.ea_expand_training:
            self.args.log_path = self.args.log_path + '-ea_expand_training'
        if self.args.RPG:
            self.args.log_path = self.args.log_path + '-RPG'
            if not self.args.use_attn:
                self.args.log_path=self.args.log_path + '-wo attn'
            if not self.args.use_RPG_triple:
                self.args.log_path=self.args.log_path + '-wo triple'
            if not self.args.use_augment:
                self.args.log_path=self.args.log_path + '-wo augment'

        self.args.log_path = self.args.log_path + '-' + str(self.args.ea_rate) + '--' +str(self.args.learning_rate) + '-' + str(args.seed) + '-neg_ratio-' + str(args.neg_ratio)
        '''add additional note to log name'''
        if self.args.note != '':
            self.args.log_path = self.args.log_path + self.args.note

        '''set logger'''
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
        console_formatter = logging.Formatter('%(asctime)-8s: %(message)s')
        logging_file_name = args.log_path + '.txt'
        file_handler = logging.FileHandler(logging_file_name)
        file_handler.setFormatter(formatter)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.formatter = console_formatter
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)
        self.args.logger = logger

        '''set device'''
        torch.cuda.set_device(int(args.gpu))
        _ = torch.tensor([1]).cuda()
        self.args.device = _.device

    def save_model(self, is_best=False, name=''):
        '''
        Save trained model.
        :param is_best: If True, save it as the best model.
        After training on each snapshot, we will use the best model to evaluate.
        '''
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.model.state_dict()
        checkpoint_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint_dict['epoch_id'] = self.args.epoch

        if is_best:
            self.args.logger.info('Saving Best Model to {}/model_best.tar'.format(self.args.save_path))
            out_tar = os.path.join(self.args.save_path, 'model_best.tar')
            torch.save(checkpoint_dict, out_tar)
            if self.args.RPG and self.args.use_attn:
                atten_weight_path = os.path.join(self.args.save_path, 'attn_weight_best.npy')
                self.kg.best_attention_weight = deepcopy(self.kg.attention_weight)
                np.save(atten_weight_path, self.kg.best_attention_weight.cpu().detach().numpy())
        if name != '':
            out_tar = os.path.join(name)
            torch.save(checkpoint_dict, out_tar)

    def load_checkpoint(self, input_file):
        if os.path.isfile(os.path.join(os.getcwd(), input_file)):
            logging.info('=> loading checkpoint \'{}\''.format(os.path.join(os.getcwd(), input_file)))
            checkpoint = torch.load(os.path.join(os.getcwd(), input_file), map_location="cuda:{}".format(self.args.gpu))
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if '-wo attn' not in input_file and 'erge' not in input_file:
                attn_path = os.path.join(os.getcwd(), input_file)[:-15]
                self.kg.best_attention_weight = torch.tensor(np.load(os.path.join(attn_path, 'attn_weight_best.npy'))).to(self.args.device)
                return int(checkpoint['epoch_id']) + 1
        else:
            logging.info('=> no checkpoint found at \'{}\''.format(input_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # training control
    parser.add_argument('-dataset', dest='dataset', default='DBP-FB', help='dataset name, DBP-FB, WIKI-YAGO')
    parser.add_argument('-load_checkpoint', dest='load_checkpoint', default=None, help='./model_best.tar')

    # base setting
    parser.add_argument('-optimizer_name', dest='optimizer_name', default='Adam')
    parser.add_argument('-epoch_num', dest='epoch_num', default=1000, help='max epoch num')
    parser.add_argument('-batch_size', dest='batch_size', default=2048, help='Mini-batch size')
    parser.add_argument('-test_batch_size', dest='test_batch_size', default=100, help='Mini-batch size')
    parser.add_argument('-learning_rate', dest='learning_rate', default=0.0005)
    parser.add_argument('-emb_dim', dest='emb_dim', default=256, help='embedding dimension')
    parser.add_argument('-l2', dest='l2', default=0.0, help='optimizer l2')

    parser.add_argument('-patience', dest='patience', default=5, help='early stop step')
    parser.add_argument('-neg_ratio', dest='neg_ratio', default=256)
    parser.add_argument('-margin', dest='margin', default=9.0, help='')
    parser.add_argument('-gpu', dest='gpu', default=0)

    parser.add_argument('-encoder', dest='encoder', default='lookup', help='lookup, lookup_attn')
    parser.add_argument('-scorer', dest='scorer', default='TransE', help='')

    # for ea
    parser.add_argument('-ea_rate', dest='ea_rate', default='0.3', help='')
    parser.add_argument('-RPG', dest='RPG', default='True', help='')
    parser.add_argument('-ea_expand_training', dest='ea_expand_training', default='True', help='')

    '''Ablation Study'''
    parser.add_argument('-use_attn', dest='use_attn', default='True', help='')
    parser.add_argument('-use_RPG_triple', dest='use_RPG_triple', default='True', help='')
    parser.add_argument('-use_augment', dest='use_augment', default='True', help='')

    '''RPG'''
    parser.add_argument('-topk', dest='topk', default=3, help='')
    parser.add_argument('-lambda_1', dest='lambda_1', default=0.7, help='')
    parser.add_argument('-lambda_2', dest='lambda_2', default=0.3, help='')
    parser.add_argument('-warmup', dest='warmup', default=10)
    parser.add_argument('-RPG_update_span', dest='RPG_update_span', default=5)
    parser.add_argument('-alpha', dest='alpha', default=1.0)

    # others
    parser.add_argument('-save_path', dest='save_path', default='./checkpoint/')
    parser.add_argument('-data_path', dest='data_path', default='./dataset/data/')
    parser.add_argument('-log_path', dest='log_path', default='./logs/')
    parser.add_argument('-num_workers', dest='num_workers', default=10)
    parser.add_argument('-seed', dest='seed', default=2024)
    parser.add_argument('-valid_metrics', dest='valid_metrics', default='mrr')
    parser.add_argument('-note', dest='note', default='develop', help='The note of log file name')
    args = parser.parse_args()
    retype_parameters(args)
    same_seeds(args.seed)

    if not args.RPG:
        args.use_augment = False
        args.use_attn = False
        args.use_RPG_triple = False
    if args.use_attn:
        args.encoder = 'lookup_attn'

    args.source_list = args.dataset.split('-')
    E = experiment(args)
    E.train()
    E.test()

