''' Data Loader class for training iteration '''
import random
import numpy as np
import torch
import math
import lamp.Constants as Constants
from pdb import set_trace as stop
import utils
from os import path

def get_matrix(tgt_data,tgt_dict,t=0.4):
    _nums=np.zeros(len(tgt_dict)-4)
    for sample in tgt_data:
        for idx in sample[1:-1]:
            if idx-4>=0:
                _nums[idx-4]+=1
    _adj=np.zeros([len(tgt_dict)-4,len(tgt_dict)-4])
    for sample in tgt_data:
        sample2=sample
        for i,idx1 in enumerate(sample[1:-1]):
            for idx2 in sample2[i+1:-1]:
                if idx1!=idx2:
                    _adj[idx1-4,idx2-4]+=1
                    _adj[idx2-4,idx1-4]+=1
    _nums_d2 = _nums[:, np.newaxis]
    _nums_d1 = _nums[np.newaxis,:]
    _nums_all= pow(_nums_d2*_nums_d1,0.5)
    _adj = _adj / _nums_all
    _adj+=np.identity(len(tgt_dict)-4)
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    return torch.tensor(_adj)

def process_data(data,opt):
    label_adj_matrix = None
    if (opt.adj_matrix_lambda > 0):
        print('using heirarchy mask')
        if 'rcv1' in opt.dataset:
            label_adj_matrix = utils.get_pairwise_adj_rcv1(data['dict']['tgt'],path.join(opt.dataset,'tf_interactions.tsv'))
        else:
            label_adj_matrix = utils.get_pairwise_adj(data['dict']['tgt'],path.join(opt.dataset,'tf_interactions.tsv'))
    elif opt.label_mask == 'prior':
        print('using prior mask')
        # train_matrix = torch.zeros(len(data['train']['tgt']),len(data['dict']['tgt']))
        # for i in range(len(data['train']['tgt'])):
        #     indices = torch.from_numpy(np.array(data['train']['tgt'][i]))
        #     x = torch.zeros(len(data['dict']['tgt']))
        #     x.index_fill_(0, indices, 1)
        #     train_matrix[i] = x
        # train_matrix = train_matrix[:,4:]
        # label_adj_matrix = torch.from_numpy(np.corrcoef(train_matrix.transpose(0,1).cpu().numpy()))
        # label_adj_matrix[label_adj_matrix < 0.0] = 0
        # for i in range(label_adj_matrix.size(0)): label_adj_matrix[i,i] = 0
        # label_adj_matrix[label_adj_matrix > 0] = 1
        
        # adj_matrix = torch.eye(len(data['dict']['tgt'])-4)
        # for sample in data['train']['tgt']:
        #     sample2 = sample
        #     for i,idx1 in enumerate(sample[1:-1]):
        #         for idx2 in sample2[i+1:-1]:
        #             if idx1 != idx2:
        #                 adj_matrix[idx1-4,idx2-4] = 1
        #                 adj_matrix[idx2-4,idx1-4] = 1

        # stop()
        # adj_matrix_pre=adj_matrix>=2000
        # adj_matrix_end=adj_matrix_pre.float()+torch.eye(adj_matrix.size(0))
        # label_adj_matrix = adj_matrix
        label_adj_matrix = get_matrix(data['train']['tgt'],data['dict']['tgt'],t=0.2)


    label_vals = torch.zeros(len(data['train']['tgt']),len(data['dict']['tgt']))
    for i in range(len(data['train']['tgt'])):
        indices = torch.from_numpy(np.array(data['train']['tgt'][i]))
        x = torch.zeros(len(data['dict']['tgt']))
        x.index_fill_(0, indices, 1)
        label_vals[i] = x
    
    values,ranking = torch.sort(label_vals.sum(0),dim=0,descending=True)
    ranking_values = values[2:-2]/values[2:-2].sum()
    # mean_tf_labels = label_vals[:,4:].sum(1).mean()
    ranking = ranking.numpy().tolist()
    ranking = ranking[2:-2]
    ranking.insert(0,2)
    ranking += [0,1,3]

    for sample in data['train']['tgt']: 
        sample = sorted(sample, key=ranking.index) 
        sample=sorted(sample,key=ranking.index)
    for sample in data['valid']['tgt']: 
        sample = sorted(sample, key=ranking.index)
    for sample in data['test']['tgt']:
        sample = sorted(sample, key=ranking.index)
    opt.max_token_text_seq_len_e = data['settings'].max_text_seq_len
    opt.max_token_visual_seq_len_e = data['settings'].max_visual_seq_len
    opt.max_token_audio_seq_len_e = data['settings'].max_audio_seq_len
    opt.max_token_seq_len_d = opt.max_ar_length
    
    if opt.summarize_data:
        utils.summarize_data(data)

    if not 'sider' in opt.dataset:
        data['train']['adj'],data['valid']['adj'],data['test']['adj'] = None,None,None
    # stop()
    #========= Preparing DataLoader =========#
    train_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'],
        src_text_insts=data['train']['src-text'],
        src_visual_insts=data['train']['src-visual'],
        src_audio_insts=data['train']['src-audio'],
        adj_insts=data['train']['adj'],
        tgt_insts=data['train']['tgt'],
        batch_size=opt.batch_size,
        binary_relevance=opt.binary_relevance,
        cuda=opt.cuda,
        shuffle=True,
        drop_last=False)

    valid_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'], 
        src_text_insts=data['valid']['src-text'],
        src_visual_insts=data['valid']['src-visual'],
        src_audio_insts=data['valid']['src-audio'],
        adj_insts=data['valid']['adj'],
        tgt_insts=data['valid']['tgt'],
        batch_size=opt.test_batch_size,
        binary_relevance=opt.binary_relevance,
        shuffle=False,
        cuda=opt.cuda)

    test_data = DataLoader(
        data['dict']['src'],
        data['dict']['tgt'], 
        src_text_insts=data['test']['src-text'],
        src_visual_insts=data['test']['src-visual'],
        src_audio_insts=data['test']['src-audio'],
        adj_insts=data['test']['adj'],
        tgt_insts=data['test']['tgt'],
        batch_size=opt.test_batch_size,
        binary_relevance=opt.binary_relevance,
        shuffle=False,
        cuda=opt.cuda)

    # opt.src_com_vocab_size = train_data.src_com_vocab_size
    # opt.src_lyr_vocab_size = train_data.src_lyr_vocab_size
    opt.tgt_vocab_size = train_data.tgt_vocab_size

    if opt.binary_relevance:
        opt.tgt_vocab_size = opt.tgt_vocab_size - 4
        opt.max_ar_length = opt.tgt_vocab_size

    return train_data,valid_data,test_data,label_adj_matrix,opt


class DataLoader(object):
    ''' For data iteration '''

    def __init__(
            self, src_word2idx, tgt_word2idx,
            src_text_insts=None,src_visual_insts=None,src_audio_insts=None, adj_insts=None, tgt_insts=None,
            cuda=True, batch_size=64, shuffle=True,
            binary_relevance=False,drop_last=False):
        assert src_text_insts is not None
        assert src_visual_insts is not None
        assert src_audio_insts is not None
        assert len(src_text_insts) >= batch_size

        if tgt_insts:
            assert len(src_text_insts) == len(tgt_insts)
        if adj_insts:
            assert len(src_text_insts) == len(adj_insts)
            self._adj_insts = adj_insts
        else:
            self._adj_insts = None


        self.cuda = cuda
        # stop()
        self._n_batch = int(np.ceil(len(src_text_insts) / batch_size)) # ceil oten denotes as  up digtal 
        if drop_last:
            self._n_batch -= 1

        self._batch_size = batch_size

        self._src_text_insts = src_text_insts
        self._src_visual_insts = src_visual_insts
        self._src_audio_insts = src_audio_insts
        
        self._tgt_insts = tgt_insts


        if src_word2idx is not None :
            src_word2idx = {idx:word for word, idx in src_com_word2idx.items()}
            self.src_word2idx = src_word2idx
            self.src_word2idx = src_word2idx
            self.long_input = True
        else:
            self.src_word2idx = src_text_insts[0]
            self.long_input = False

        tgt_idx2word = {idx:word for word, idx in tgt_word2idx.items()}
        
        self._tgt_word2idx = tgt_word2idx
        self._tgt_idx2word = tgt_idx2word

        self._iter_count = 0


        self._need_shuffle = shuffle

        if self._need_shuffle:
            self.shuffle()

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_com_insts)

    @property
    def src_com_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_com_word2idx)
    @property
    def src_lyr_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._src_lyr_word2idx)

    @property
    def tgt_vocab_size(self):
        ''' Property for vocab size '''
        return len(self._tgt_word2idx)

    @property
    def src_com_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_com_word2idx
    @property
    def src_lyr_word2idx(self):
        ''' Property for word dictionary '''
        return self._src_lyr_word2idx

    @property
    def tgt_word2idx(self):
        ''' Property for word dictionary '''
        return self._tgt_word2idx

    @property
    def src_com_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_com_idx2word
    @property
    def src_lyr_idx2word(self):
        ''' Property for index dictionary '''
        return self._src_lyr_idx2word

    @property
    def tgt_idx2word(self):
        ''' Property for index dictionary '''
        return self._tgt_idx2word

    def shuffle(self):
        ''' Shuffle data for a brand new start '''
        if self._tgt_insts and self._adj_insts:
            paired_insts = list(zip(self._src_com_insts,self._src_lyr_insts, self._adj_insts,self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_com_insts,self._src_lyr_insts, self._adj_insts, self._tgt_insts = zip(*paired_insts)
        elif self._tgt_insts:
            paired_insts = list(zip(self._src_text_insts,self._src_visual_insts,self._src_audio_insts,self._tgt_insts))
            random.shuffle(paired_insts)
            self._src_text_insts,self._src_visual_insts,self._src_audio_insts, self._tgt_insts = zip(*paired_insts)
        else:
            paired_insts = list(zip(self._src_com_insts,self._src_lyr_insts))
            random.shuffle(paired_insts)
            self._src_com_insts,self._src_lyr_insts = zip(*paired_insts)
            # random.shuffle(self._src_com_insts,self._src_com_insts)


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''

        def convert_string_to_mat(adj_string):
            dim = int(math.sqrt(len(adj_string)))

            output = torch.Tensor(adj_string).view(dim,dim)#.type(torch.uint8)

            if self.cuda:
                output = output.cuda()

            return(output)

        def construct_adj_mat(insts,encoder=False):

            inst_data_tensor = [convert_string_to_mat(inst) for inst in insts]

            return inst_data_tensor

        def pad_src_to_longest(insts,encoder=False):
            ''' Pad the instance to the max seq length in batch '''
            # max_len = max(len(line) for inst in insts for line in inst)
            # max_seq_len=max(len(inst) for inst in insts)
            # # stop()
            # pad_line=np.zeros(max_len).tolist()
            # inst_data = np.array([[
            #     line + [Constants.PAD] * (max_len - len(line))
            #     for line in inst ]+[pad_line] * (max_seq_len - len(inst)) for inst in insts])

            # inst_position = np.array([[
            #     [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(line)]
            #     for line in inst ]for inst in inst_data])


            # inst_data_tensor = torch.Tensor(inst_data)
            # inst_position_tensor = torch.Tensor(inst_position)
            # if self.cuda:
            #     inst_data_tensor = inst_data_tensor.cuda()
            #     inst_position_tensor = inst_position_tensor.cuda()
            
            inst_position=np.array([ [pos_i+1  for pos_i, w_i in enumerate(line)] for line in insts])
            inst_data_tensor = torch.Tensor(insts)
            inst_position_tensor = torch.Tensor(inst_position)
            return inst_data_tensor, inst_position_tensor
        def pad_tgt_to_longest(insts,encoder=False):
            ''' Pad the instance to the max seq length in batch '''
            max_len = max(len(inst) for inst in insts)

            inst_data = np.array([
                inst + [Constants.PAD] * (max_len - len(inst))
                for inst in insts])

            inst_position = np.array([
                [pos_i+1 if w_i != Constants.PAD else 0 for pos_i, w_i in enumerate(inst)]
                for inst in inst_data])


            inst_data_tensor = torch.Tensor(inst_data)
            inst_position_tensor = torch.Tensor(inst_position)
            if self.cuda:
                inst_data_tensor = inst_data_tensor.cuda()
                inst_position_tensor = inst_position_tensor.cuda()
                
            return inst_data_tensor, inst_position_tensor

        if self._iter_count < self._n_batch:
            batch_idx = self._iter_count
            self._iter_count += 1

            start_idx = batch_idx * self._batch_size
            end_idx = (batch_idx + 1) * self._batch_size

            src_text_insts = self._src_text_insts[start_idx:end_idx]
            src_visual_insts = self._src_visual_insts[start_idx:end_idx]
            src_audio_insts = self._src_audio_insts[start_idx:end_idx]

            if self._adj_insts:
                adj_insts = construct_adj_mat(self._adj_insts[start_idx:end_idx])
            else:
                adj_insts = None

            src_text_data, src_text_pos = pad_src_to_longest(src_text_insts,encoder=True)
            src_visual_data, src_visual_pos = pad_src_to_longest(src_visual_insts,encoder=True)
            src_audio_data, src_audio_pos = pad_src_to_longest(src_audio_insts,encoder=True)

            src_text_pos = src_text_pos.long()
            src_visual_pos = src_visual_pos.long()
            src_audio_pos = src_audio_pos.long()
            
            if self.long_input:
                src_com_data = src_com_data.long()
                src_lyr_data = src_lyr_data.long()
                

            # stop()
            if not self._tgt_insts:
                return src_com_data, src_lyr_pos,src_lyr_data,src_lyr_pos
            else:
                tgt_insts = self._tgt_insts[start_idx:end_idx]
                tgt_data, tgt_pos = pad_tgt_to_longest(tgt_insts)
                tgt_data = tgt_data.long()
                tgt_pos = tgt_pos.long()
                return (src_text_data, src_text_pos,src_visual_data,src_visual_pos,src_audio_data,src_audio_pos), (adj_insts), tgt_data

        else:

            if self._need_shuffle:
                self.shuffle()

            self._iter_count = 0
            raise StopIteration()