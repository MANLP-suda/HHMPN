import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lamp.Constants as Constants
from lamp.Layers import EncoderLayer,DecoderLayer
from lamp.SubLayers import ScaledDotProductAttention
from lamp.SubLayers import PositionwiseFeedForward
from lamp.SubLayers import XavierLinear
from lamp.SubLayers import NodeAttention
from lamp.SubLayers import MultiAngleFusion
from pdb import set_trace as stop 
from lamp import utils
import copy



class RNNDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1):
        # dropout = 0

        super(RNNDecoder, self).__init__()
        n_position = n_max_seq + 1
        self.n_max_seq = n_max_seq
        self.d_model = d_model
        self.n_tgt_vocab = n_tgt_vocab
        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.dropout = nn.Dropout(dropout)
        self.attention_stack = nn.ModuleList([ScaledDotProductAttention(d_model,dropout=dropout)for _ in range(n_layers)])
        self.rnn_layer_stack = nn.ModuleList([
            nn.GRU(d_model+d_word_vec,d_model,batch_first=True,dropout=dropout)
            for _ in range(n_layers)])
        self.U = nn.Linear(self.d_model, self.n_tgt_vocab)
        self.V = nn.Linear(self.d_model, self.n_tgt_vocab)
        self.C = nn.Linear(self.d_model, self.n_tgt_vocab)

    def forward_step(self, input_var, decoder_hidden, encoder_outputs,dec_enc_attn_pad_mask=None):
        batch_size = input_var.size(0)
        embedded = self.tgt_word_emb(input_var)
        decoder_hidden = decoder_hidden.view(batch_size,1,-1)
        
        if encoder_outputs.size(1) == 1:
            dec_enc_attn_pad_mask=None

        for idx,dec_layer in enumerate(self.rnn_layer_stack):
            context,attn = self.attention_stack[idx](decoder_hidden.view(batch_size,1,-1),encoder_outputs,encoder_outputs,dec_enc_attn_pad_mask)
            rnn_input = torch.cat((embedded,context),2)
            embedded,decoder_hidden = dec_layer(rnn_input, decoder_hidden.view(1,batch_size,-1))

        output = self.U(decoder_hidden)
        output += self.V(embedded.view(batch_size,-1))
        output += self.C(context.view(batch_size,-1))

        return output, decoder_hidden, attn

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        batch_size = enc_output.size(0)
        dec_enc_attn_pad_mask = utils.get_attn_padding_mask(tgt_seq, src_seq,unsqueeze=False)
        dec_output = torch.zeros(tgt_seq.size(0),tgt_seq.size(1),self.n_tgt_vocab).cuda()
        dec_input = tgt_seq[:,0].unsqueeze(1)
        decoder_hidden = enc_output.mean(1)

        for di in range(tgt_seq.size(1)):
            decoder_output,decoder_hidden,step_attn=self.forward_step(dec_input,decoder_hidden,enc_output,dec_enc_attn_pad_mask)

            dec_output[:,di,:] = decoder_output
            dec_input = F.log_softmax(decoder_output.view(batch_size,-1),dim=1).topk(1)[1].view(batch_size,-1)

        return dec_output,


class MLPDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''
    def __init__(
            self, n_tgt_vocab, n_max_seq_e, n_max_seq_d, n_layers=6, n_head=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,enc_transform='mean'):
        super(MLPDecoder, self).__init__()
        self.n_max_seq = n_max_seq_e
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.enc_transform = enc_transform
        if enc_transform in ['flatten']: raise NotImplementedError
        self.linear1 = nn.Linear(d_model,d_model)
        self.linear4 = nn.Linear(d_model,n_tgt_vocab)

    def forward(self, tgt_seq, src_seq, enc_output,return_attns=False,int_preds=False):
        batch_size = src_seq.size(0)
        x = enc_output.float()

        out1 = self.dropout(F.relu(self.linear1(x)))
        dec_dropout = self.linear4(out1)
        return dec_dropout.view(batch_size,1,-1),
    

class GraphDecoder(nn.Module):
    def __init__(
            self, n_tgt_vocab, n_max_seq, n_layers=6, n_head=8,n_head2=8, d_k=64, d_v=64,
            d_word_vec=512, d_model=512, d_inner_hid=1024, dropout=0.1,dropout2=0.1,
            no_dec_self_att=False,label_adj_matrix=None,label_mask=None,
            enc_vec=True,graph_conv=False,attn_type='softmax'):
        super(GraphDecoder, self).__init__()
        self.enc_vec = enc_vec
        self.dropout = nn.Dropout(dropout)
        self.constant_input = torch.from_numpy(np.arange(n_tgt_vocab)).view(-1,1)

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        # stop()
        if label_adj_matrix is not None:
            for i in range(label_adj_matrix.size(0)):
                if label_adj_matrix[i].sum().item() < 1:
                    label_adj_matrix[i,i] = 1 #This prevents Nan output in attention (otherwise 0 attn weights occurs)
            self.label_mask = utils.swap_0_1(label_adj_matrix,1,0).unsqueeze(0)
        else:
            if label_mask == 'inveye':
                self.label_mask = 1-torch.eye(n_tgt_vocab)
            elif label_mask == 'none':
                self.label_mask = None
            else:
                NotImplementedError
        
        self.layer_text_stack = nn.ModuleList()
        for _ in range(n_layers):
            self.layer_text_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,dropout2=dropout2,no_dec_self_att=True,attn_type=attn_type))           
        self.layer_visual_stack = nn.ModuleList()
        for _ in range(n_layers):
            self.layer_visual_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,dropout2=dropout2,no_dec_self_att=True,attn_type=attn_type))           
        self.layer_audio_stack = nn.ModuleList()
        for _ in range(n_layers):
            self.layer_audio_stack.append(DecoderLayer(d_model, d_inner_hid, n_head,n_head2, d_k, d_v, dropout=dropout,dropout2=dropout2,no_dec_self_att=True,attn_type=attn_type))           
        self.com_nodeattn=NodeAttention(d_model)
        self.lyr_nodeattn=NodeAttention(d_model)
        self.multi_angle_fus=MultiAngleFusion(d_model)
        self.proj1 = nn.Linear(1536, 1024)
        self.proj2 = nn.Linear(1024, 512)
        self.dense=nn.Linear(512,6)

    def forward(self, tgt, src_text_seq, enc_text_output,src_visual_seq, enc_visual_output,src_audio_seq, enc_audio_output,return_attns=False, int_preds=False):
        '''
        @description: 
        @param {tgt}: 
        @return: 
        '''
        # stop()
        batch_size,seq_len,dim = enc_text_output.shape
        
        if int_preds: int_outs = []
        if return_attns: dec_slf_attns, dec_enc_attns = [], []
        #contrast to traditional transformer decoder,the input in here is all the tgt
        tgt_seq= self.constant_input.repeat(1,batch_size).transpose(0,1).cuda()
        # stop()
        dec_input = self.tgt_word_emb(tgt_seq)

        dec_enc_text_attn_pad_mask = None
        dec_enc_visual_attn_pad_mask = None
        dec_enc_audio_attn_pad_mask = None
        # stop()
        if not self.enc_vec:
            #[B,L,Word_len]  denoting that a mask is need to the src for each label 
            dec_enc_text_attn_pad_mask = utils.get_attn_tgt_src_mask(tgt_seq, src_text_seq).cuda()
            dec_enc_visual_attn_pad_mask = utils.get_attn_tgt_src_mask(tgt_seq, src_visual_seq).cuda()
            dec_enc_audio_attn_pad_mask = utils.get_attn_tgt_src_mask(tgt_seq, src_audio_seq).cuda()

        if self.label_mask is not None:
            dec_slf_attn_mask = self.label_mask.repeat(batch_size,1,1).cuda().byte()
            # dec_visual_slf_attn_mask = self.label_mask.repeat(batch_size,1,1).cuda().byte()
            # dec_audio_slf_attn_mask = self.label_mask.repeat(batch_size,1,1).cuda().byte()
        else:
            dec_slf_attn_mask = None
        # stop()    
        dec_text_output= dec_input
        dec_visual_output = dec_input
        dec_audio_output = dec_input
        for idx,dec_layer in enumerate(self.layer_text_stack):
            dec_text_output, dec_text_output_int, dec_text_slf_attn, dec_text_enc_attn = dec_layer(dec_text_output, enc_text_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_text_attn_pad_mask)

            if int_preds:
                if dec_text_output_int is not None:
                    int_text_outs += [dec_text_output_int]
                int_text_outs += [dec_text_output]

            if return_attns:
                dec_text_slf_attns += [dec_text_slf_attn]
                dec_text_enc_attns += [dec_text_enc_attn]
        for idx,dec_layer in enumerate(self.layer_visual_stack):
            dec_visual_output, dec_visual_output_int, dec_visual_slf_attn, dec_visual_enc_attn = dec_layer(dec_visual_output, enc_visual_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_visual_attn_pad_mask)

            if int_preds:
                if dec_visual_output_int is not None:
                    int_visual_outs += [dec_visual_output_int]
                int_visual_outs += [dec_visual_output]

            if return_attns:
                dec_visual_slf_attns += [dec_visual_slf_attn]
                dec_visual_enc_attns += [dec_visual_enc_attn]
        for idx,dec_layer in enumerate(self.layer_audio_stack):
            dec_audio_output, dec_audio_output_int, dec_audio_slf_attn, dec_audio_enc_attn = dec_layer(dec_audio_output, enc_audio_output,slf_attn_mask=dec_slf_attn_mask,dec_enc_attn_mask=dec_enc_audio_attn_pad_mask)

            if int_preds:
                if dec_audio_output_int is not None:
                    int_audio_outs += [dec_audio_output_int]
                int_audio_outs += [dec_audio_output]

            if return_attns:
                dec_audio_slf_attns += [dec_audio_slf_attn]
                dec_audio_enc_attns += [dec_audio_enc_attn]

        # stop()
        dec_output=self.multi_angle_fus(dec_text_output,dec_visual_output,dec_audio_output,type='angleFus')
        #self_attn
        # stop()
        # dec_output_cat=torch.cat([enc_text_output[:,-1,:],enc_visual_output[:,-1,:],enc_audio_output[:,-1,:]],dim=-1)
        # dec_output_cat = self.proj2(F.dropout(F.relu(self.proj1(dec_output_cat)), 0.1))
        # dec_output=self.dense(dec_output_cat)

        if int_preds:
            return dec_com_output, int_com_outs
        elif return_attns:
            return dec_output, dec_slf_attns, dec_enc_attns
        else:
            return dec_output, None         
