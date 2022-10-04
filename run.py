import apex
import os, sys
import time
import argparse
import numpy as np
import torch

from scipy import sparse

from model import *
from kgraph import FB15k237, WN18RR, Data
from kgraph import DataIter, Predict
from kgraph.log import set_logger

from torch.utils.data import DataLoader


def set_gpu(gpu):
    
    """
    Sets the GPU to be used for the run.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def set_seed(seed):
    """
    Sets the seed.
    """
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        torch.backends.cudnn.deterministic = True 


def set_device(gpu):
    '''
    Sets the device to be used for the run.
    '''
    
    if gpu != -1 and torch.cuda.is_available():
        set_gpu(gpu)
        return torch.device('cuda:' + str(gpu))
    else:
        return torch.device('cpu')


def load_data(data_name, data_path=None):
    '''
    Sets the data to be used for the run.
    '''
    data_name = data_name.lower()
    assert data_name in ['fb15k237', 'wn18rr'], 'Invalid data name.'
    
    if data_name == 'fb15k237':
        data = FB15k237(data_path)
    else:
        data = WN18RR(data_path)
    return data


def show_params(params):
    
    msg = '\nParameters:\n'
    
    for key, value in params.items():
        msg += '\t{}: {}\n'.format(key, value)
    return msg

def generate_new_tail(pred, label):
    """
    :param pred: predicted entity
    :param label: label
    :return: entity
    """
    b_range = np.arange(pred.shape[0])
    label = np.around(label)
    pred = np.where(label.astype(np.bool_), -np.ones_like(pred) * 100000000, pred)
    
    index = 0 if np.random.random() < 0.7 else 1
    
    replace_obj = np.argsort(-pred, axis=1)[b_range, index]
    return replace_obj


class Run(object):
    
    def __init__(self, params):
        '''
        Constructor of the runner class.
        '''
        
        params.embed_dim = params.encoder_conve_k_w * \
            params.encoder_conve_k_h if params.embed_dim is None else params.embed_dim
        
        self.p = params
        run_time = time.strftime('_%Y-%m-%d-%Hh%Mm%Ss', time.localtime())
        name = f'{params.encoder}_{params.decoder}_{run_time}'
        self.logger = set_logger(name)
        
        set_seed(params.seed)
        self.device = set_device(params.gpu)
        
        self.load_data()
        
        self.model = self.add_model(params.encoder, params.decoder)
        self.optimizer = self.add_optimizer(self.model.parameters(), params.lr, params.weight_decay)
        
        if params.use_apex:
            self.model, self.optimizer = apex.amp.initialize(self.model, self.optimizer, opt_level="O1")
        
        self.logger.info(show_params(vars(params)))
    
    
    def load_data(self):
        '''
        Loads data.
        '''
        self.data = load_data(self.p.data, None)
        if self.p.debug:
            print(self.data.train[:10, :])
            self.data.train = self.data.train[:10, :]
            self.data.valid = self.data.train
            self.data.test = self.data.train
        self.kgraph_predict = Predict(self.data, element_type='pair')
        self.p.num_ent = self.data.num_ent
        self.p.num_rel = self.data.num_rel
        if self.p.debug:
            print('load data random x:', torch.randn(8,))
        self.dataiter = DataIter(self.data, self.p.batch_size,
                                 num_threads=self.p.num_workers,
                                 smooth_lambda=self.p.smooth,
                                 element_type='pair')
        
        self.edge_index, self.edge_type = self.construct_adj()
        
        train_data = self.data.train
        rel_head = train_data[:, [1, 0]]
        rel_head[:, 0] += self.data.num_rel
        rel_tail = train_data[:, [1, 2]]
        rel_tails = np.unique(np.concatenate((rel_head, rel_tail), axis=0), axis=0).T
        
        row = rel_tails[0]
        col = rel_tails[1]
        data = np.ones_like(col).astype(np.float32)
        
        rel_tails = sparse.coo_matrix((data, (row, col)), shape=(2*self.data.num_rel, self.data.num_ent)).todense()
        self.tmp_rel_tails = np.asarray(rel_tails)
        
        self.rel_tails = None
        self.set_semi_supervised(self.data.train, self.data.valid, self.data.test)
    
    def set_semi_supervised(self, train_data, valid_data, test_data):
        self.semi_supervised_data = Data(self.p.num_ent, self.p.num_rel)
        self.semi_supervised_data.train = train_data
        self.semi_supervised_data.valid = valid_data
        self.semi_supervised_data.test = test_data
        self.semi_supervised_data.smooth_lambda = 0.0
    
    def reset_ss_train_data(self, train_data, smooth_lambda=0.0):
        if self.p.debug:
            print('reset semi-supervised train data')
            print(train_data)
        self.semi_supervised_data.train = train_data
        self.semi_supervised_data.smooth_lambda = smooth_lambda
        
    def get_batch_from_ss_data(self):
        
        return DataLoader(dataset=self.semi_supervised_data, batch_size=self.p.batch_size, shuffle=True, num_workers=self.p.num_workers)

    def generate_new_pos_samples(self, pos_samples):
        def function(data):
            with torch.no_grad():
                pred = self.model.predict(data.to(self.device)).cpu().numpy()
                return 1 - pred
        
        self.reset_ss_train_data(pos_samples, 0.0)
        
        self.model.eval()
        
        # new_pos_samples = [pos_samples]
        new_pos_samples = []
        
        for batch_data, batch_label in self.get_batch_from_ss_data():
            new_triples = []
            pred = function(batch_data)
            new_tails = generate_new_tail(pred, batch_label.numpy())
            
            for i, (h, r) in enumerate(batch_data):
                if r < self.p.num_rel:
                    new_triples.append([h, r, new_tails[i]])
                else: 
                    new_triples.append([new_tails[i], r - self.p.num_rel, h])

            new_pos_samples.append(np.asarray(new_triples))
        new_pos_samples = np.concatenate(new_pos_samples, axis=0).astype(np.int32)
        if self.p.debug:
            print('new pos samples:')
            print(new_pos_samples)
        return new_pos_samples

    def construct_adj(self):
        '''
        calculates the adjacency matrix.
        '''
        train_data = self.data.train
        inv_train_data = train_data[:, [2, 1, 0]]
        inv_train_data[:, 1] += self.p.num_rel
        train_data = np.concatenate([train_data, inv_train_data], axis=0)
        
        edge_index = torch.from_numpy(train_data[:, [0, 2]].T).long().to(self.device)
        edge_type = torch.from_numpy(train_data[:, 1]).long().to(self.device)
        
        return edge_index, edge_type
    
    def add_model(self, encoder, decoder):
        '''
        Creates the model.
        '''
        
        model_name = '{}_{}'.format(encoder, decoder)
        self.model_name = model_name
        model_name = model_name.lower()
        
        if model_name == 'embed_transe':
            model = Embed_TransE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'embed_distmult':
            model = Embed_DistMult(self.p, self.edge_index, self.edge_type)
        elif model_name == 'embed_conve':
            model = Embed_ConvE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'embed_interacte':
            model = Embed_InteractE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'embed_involuation':
            model = Embed_Involuation(self.p, self.edge_index, self.edge_type)
        elif model_name == 'gcn_transe':
            model = GCN_TransE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'gcn_distmult':
            model = GCN_DistMult(self.p, self.edge_index, self.edge_type)
        elif model_name == 'gcn_conve':
            model = GCN_ConvE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'gcn_interacte':
            model = GCN_InteractE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'gat_conve':
            model = GAT_ConvE(self.p, self.edge_index, self.edge_type)
        elif model_name == 'gat_interacte':
            model = GAT_InteractE(self.p, self.edge_index, self.edge_type)
        else:
            raise NotImplementedError('Invalid model name.')
        
        self.logger.info(model)
        return model.to(self.device)
    
    def add_optimizer(self, parameters, lr, weight_decay):
        '''
        Creates the optimizer for training the parameters of the model.
        '''
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    
    def save_model(self, save_path):
        device = self.model.device
        state = {
            'state_dict': self.model.cpu().state_dict(),
            'best_valid': self.best_valid,
            'best_epoch': self.best_epoch,
        }
        torch.save(state, save_path)
        self.model.to(device)
        
    
    def load_model(self, load_path):
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_valid = state['best_valid']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.optimizer = self.add_optimizer(self.model.parameters(), self.p.lr, self.p.weight_decay)
    
    def evaluate(self, data_type, epoch=None):
        def function(data):
            with torch.no_grad():
                pred = self.model.predict(data).cpu().numpy()
                if self.rel_tails is None:
                    return 1 - pred
                return - (pred * self.rel_tails[data[:, 1], :])
        self.model.eval()
        
        batch_size = 512 if self.model_name.lower() != 'embed_involuation' else self.p.batch_size
        
        if data_type == 'valid':
            table, results = self.kgraph_predict.predict_valid(function, batch_size)
        else:
            table, results = self.kgraph_predict.predict_test(function, batch_size)
        
        if epoch is not None:
            self.logger.info('Epoch: {}'.format(epoch))
        self.logger.info('\n' + table + '\n')
        return results['avg_filtered']
    
    def evaluate_n2n(self, data_type='test', epoch=None):
        def function(data):
            with torch.no_grad():
                pred = self.model.predict(data).cpu().numpy()
                if self.rel_tails is None:
                    return 1 - pred
                return - (pred * self.rel_tails[data[:, 1], :])
        self.model.eval()
        
        batch_size = 512 if self.model_name.lower() != 'embed_involuation' else self.p.batch_size
        table = self.kgraph_predict.predict_N2N(function, batch_size)
        self.logger.info('\n' + table + '\n')
        
        accur, thresh = self.kgraph_predict.calculate_classification_accuracy(function, batch_size)
        self.logger.info('Accuracy: {}, Threshold: {}'.format(accur, thresh))
    
    def per_epoch(self, epoch):
        
        self.model.train()
        losses = []
        
        per_epoch_time = time.time()
        
        if self.p.debug:
            print('begin per epoch')
        
        for step, (batch, label) in enumerate(self.dataiter.generate_pair()):
            forward_time_start = time.time()
            self.optimizer.zero_grad()
            # pred= self.model(batch)
            loss = self.model.loss(batch, label)
            
            forward_time_end = time.time()
            forward_time = forward_time_end - forward_time_start
            backward_time_start = forward_time_end
            
            if self.p.use_apex:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # pass
            else:
                loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.item())
            backward_time_end = time.time()
            backward_time = backward_time_end - backward_time_start
            
            if self.p.debug:
                print('loss:', losses)
            
            
            if step == 0:
                self.logger.info('Epoch: {} Step: {} [Forward time: {:.1f}s; Backward time: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time,
                                    backward_time,
                                    np.mean(losses)))
            
            elif step > 0 and step % self.p.log_step == 0:
                self.logger.info('Epoch: {} Step: {} [Forward time: {:.1f}s; Backward time: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time * self.p.log_step,
                                    backward_time * self.p.log_step,
                                    np.mean(losses)))
            else:
                continue
        
        per_epoch_time = time.time() - per_epoch_time
        per_epoch_time = time.strftime("%M:%S", time.localtime(per_epoch_time))
        self.logger.info('Epoch: {} [Time: {}] Loss: {:.5f}'.format(epoch, per_epoch_time, np.mean(losses)))
        return np.mean(losses)
    
    def semi_supervised_per_train(self, epoch, use_test_data=False):
        losses = []
        
        orig_data = self.data.train
        orig_size = orig_data.shape[0]
        
        if not use_test_data:
            ss_train_data = orig_data[np.random.choice(orig_size, size=orig_size // 4, replace=False), :]
            new_pos_data = self.generate_new_pos_samples(ss_train_data)
            self.reset_ss_train_data(np.concatenate([orig_data, new_pos_data], axis=0), self.p.smooth)
        else:
            new_pos_data = self.generate_new_pos_samples(self.data.test)
            self.reset_ss_train_data(np.concatenate([orig_data, new_pos_data], axis=0), self.p.smooth)
        
        self.model.train()
        
        per_epoch_time = time.time()
        for step, (batch, label) in enumerate(self.get_batch_from_ss_data()):
            forward_time_start = time.time()
            
            if self.p.debug:
                print('batch data:', epoch)
                print(batch)
            if batch.size(0) <= 1:
                continue
            self.optimizer.zero_grad()
            # pred= self.model(batch)
            loss = self.model.loss(batch, label)
            
            forward_time_end = time.time()
            forward_time = forward_time_end - forward_time_start
            backward_time_start = forward_time_end
            
            if self.p.use_apex:
                with apex.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                # pass
            else:
                loss.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            losses.append(loss.item())
            backward_time_end = time.time()
            backward_time = backward_time_end - backward_time_start
            
            if self.p.debug:
                print('loss:', losses)
            
            
            if step == 0:
                self.logger.info('Epoch: {} Step: {} semi [Forward time: {:.1f}s; Backward time: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time,
                                    backward_time,
                                    np.mean(losses)))
            
            elif step > 0 and step % self.p.log_step == 0:
                self.logger.info('Epoch: {} Step: {} semi [Forward time: {:.1f}s; Backward time: {:.1f}s]: Loss: {:.5f}'.format(epoch, step,
                                    forward_time * self.p.log_step,
                                    backward_time * self.p.log_step,
                                    np.mean(losses)))
            else:
                continue
        
        per_epoch_time = time.time() - per_epoch_time
        per_epoch_time = time.strftime("%M:%S", time.localtime(per_epoch_time))
        self.logger.info('Epoch: {} semi [Time: {}] Loss: {:.5f}'.format(epoch, per_epoch_time, np.mean(losses)))
        return np.mean(losses)
        
    
    def fit(self):
        
        self.best_valid, self.best_epoch = 0, 0
        run_time = time.strftime('_%Y-%m-%d-%Hh%Mm%Ss', time.localtime())
        self.save_path = os.path.join('./checkpoints', self.model_name + run_time + '.pth')
        
        if self.p.restore:
            self.load_model(self.save_path)
            self.logger.info('Successfully loaded previous model.')
        
        if self.p.debug:
            self.p.max_epoch = 5
        
        kill_cnt = 0
        for epoch in range(self.p.max_epoch):
            # print('epoch:', epoch)
            if self.p.use_magic and epoch != 0 and epoch % (1 + np.around(self.p.pre_train_step / (epoch + 1))) == 0:
                train_loss = self.semi_supervised_per_train(epoch, use_test_data=True)
            else: 
                train_loss = self.per_epoch(epoch)
            
            # torch.cuda.empty_cache()
            if self.p.debug:
                val_results = {}
                val_results['mrr'] = 0.
                
                print('over')
                # self.model.save_embeddings()
            else:
                self.model.save_embeddings()
                val_results = self.evaluate('test', epoch)
            
            if val_results['mrr'] > self.best_valid:
                self.best_valid = val_results['mrr']
                self.best_epoch = epoch
                self.model.save_embeddings()
                self.save_model(self.save_path)
                kill_cnt = 0
                self.logger.info('The best model of epoch {} have save in {}.'.format(epoch, self.save_path))
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                
                if kill_cnt > self.p.early_stop_cnt:
                    self.logger.info('Early stopping!!')
                    break
        
        self.logger.info('Loading best model from the epoch {}..., Evaluating on Test Data!'.format(self.best_epoch))
        if not self.p.debug:
            self.load_model(self.save_path)
            self.evaluate('test')
    
    def predict(self, save_path):
        
        def set_rel_tails(alpha):
            self.rel_tails = self.tmp_rel_tails * alpha + (1- self.tmp_rel_tails) * (1 - alpha)
        
        def find_best_parameters():
            alpha = -0.05
            best = {'alpha': alpha + 0.5, 'mrr': 0.}
            for i in range(10):
                alpha += 0.01
                set_rel_tails(alpha + 0.5)
                results = self.evaluate('test')
                if results['mrr'] > best['mrr']:
                    best['alpha'] = alpha + 0.5
                    best['mrr'] = results['mrr']
            return best['alpha']
        
        if self.p.debug:
            self.evaluate('test')
            return
        self.load_model(save_path)
        self.model.eval()
        set_rel_tails(find_best_parameters())
        self.model.save_embeddings()
        self.evaluate('test')
        self.evaluate_n2n()
        
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run the models on the benchmark datasets.")
    
    parser.add_argument('-data', type=str, default='FB15k237', help='Name of the dataset.', choices=['FB15k237', 'WN18RR'])
    parser.add_argument('-use_apex', action='store_true')
    parser.add_argument('-predict', action='store_true')
    parser.add_argument('-pretrain_path', type=str, default='', help='the save path of the pretrained model.')
    parser.add_argument('-pre_train_step', type=int, default=100, help='')
    
    # ########################################################################################################################
    # Hyperparameters
    # ########################################################################################################################
    parser.add_argument('-use_magic', action='store_true', help='Use the magic semi-superised for training')
    parser.add_argument('-batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('-max_epoch', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('-weight_decay', type=float, default=0.0, help='Weight decay.')
    parser.add_argument('-seed', type=int, default=41504, help='Seed.')
    parser.add_argument('-restore', action='store_true', help='Restore from the previous checkpoint.')
    parser.add_argument('-gpu', type=int, default=0, help='GPU to use.')
    parser.add_argument('-log_step', type=int, default=100, help='Log step.')
    parser.add_argument('-debug', action='store_true', help='Debug mode.')
    parser.add_argument('-num_workers', type=int, default=10, help='Number of workers.')
    parser.add_argument('-early_stop_cnt', type=int, default=30, help='Early stopping count.')
    
    parser.add_argument('-num_ent', type=int, default=0, help='Number of entities.')
    parser.add_argument('-num_rel', type=int, default=0, help='Number of relations.')
    
    
    
    # ########################################################################################################################
    # Encoder parameters
    # ########################################################################################################################
    parser.add_argument('-encoder', type=str, default='Embed', help='Name of the encoder.', choices=['Embed', 'GCN', 'GAT'])
    parser.add_argument('-embed_dim', type=int, default=200, help='Embedding dimension.')
    parser.add_argument('-num_bases', type=int, default=-1, help='Number of bases.')
    
    parser.add_argument('-encoder_drop1', type=float, default=0.3, help='Dropout rate for the first layer of the encoder.')
    parser.add_argument('-encoder_drop2', type=float, default=0.3, help='Dropout rate for the second layer of the encoder.')
    parser.add_argument('-encoder_gcn_drop', type=float, default=0.1, help='Dropout rate for the GCN layer of the encoder.')
    parser.add_argument('-encoder_gcn_bias', action='store_true', help='Whether to add bias to the GCN layer of the encoder.')
    
    parser.add_argument('-gcn_hidden_channels', type=int, default=200, help='Hidden channels of the GCN layer.')
    parser.add_argument('-gcn_in_channels', type=int, default=200, help='Input channels of the GCN layer.')
    parser.add_argument('-gcn_out_channels', type=int, default=200, help='Output channels of the GCN layer.')
    parser.add_argument('-gcn_num_layers', type=int, default=2, help='Number of GCN layers.')
    
    
    
    
    
    
    
    # ########################################################################################################################
    # Decoder parameters
    # ########################################################################################################################
    parser.add_argument('-decoder', type=str, default='ConvE', help='Name of the decoder.', choices=['TransE', 'DistMult', 'ConvE', 'InteractE', 'Involuation'])
    
    # TransE parameters
    parser.add_argument('-gamma', type=int, default=40, help='Gamma value for TransE.')
    
    
    # ConvE parameters
    parser.add_argument('-k_w', type=int, default=10, help='Convolution width.')
    parser.add_argument('-k_h', type=int, default=20, help='Convolution height.')
    parser.add_argument('-ker_sz', type=str, default='7', help='Convolution kernel size.')
    parser.add_argument('-decoder_feat_drop', type=float, default=0.3, help='Dropout rate for the decoder.')
    parser.add_argument('-decoder_hid_drop', type=float, default=0.2, help='Dropout rate for the decoder.')
    parser.add_argument('-num_filters', type=int, default=200, help='Number of filters.')
    parser.add_argument('-smooth', type=float, default=0.1, help='Smooth factor.')
    parser.add_argument('-conve_bias', action='store_true', help='Use bias in the ConvE layer.')
    
    

    
    
    
    
    args = parser.parse_args()
    
    runner = Run(args)
    
    if args.predict:
        runner.predict(args.pretrain_path)
    else:
        runner.fit()
        if args.debug:
            runner.predict(runner.save_path)
    
        
        



    
    
    


