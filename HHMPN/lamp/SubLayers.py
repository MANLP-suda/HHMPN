import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.init as init
from pdb import set_trace as stop
import numpy as np


class XavierLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True):
        super(XavierLinear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight) #parm meter initialize
    def forward(self, x):
        return self.linear(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None,stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()


        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False,attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k,bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k,bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v,bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5),attn_type=attn_type,dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model,bias=False)
            nn.init.xavier_normal_(self.fc.weight)


    def forward(self, q, k, v, attn_mask=None,dec_self=False): 

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        
        if hasattr(self,'dropout2'):
            q = self.dropout2(q)

        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)


        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv


        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1) # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        if hasattr(self,'fc'):
            output = self.fc(output)

        if hasattr(self,'dropout'):
            output = self.dropout(output)
        

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)

        
        output = self.layer_norm(output + residual)
        return output


class NodeAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.get_score_1=nn.Linear(dim,32)
        self.get_score_2=nn.Linear(32,1)


    def forward(self,x,type='nodeAttn'):#[B,seq_len,tgt_vocab,dim]
        if type=='nodeAttn':
            B,seq_len,tgt_vocab,dim=x.shape
            x_t=x.transpose(1,2).contiguous()#[B,tgt_vocab,seq_len,dim]
            x_score=self.get_score_2(self.get_score_1(x_t))#[B,tgt_vocab,seq_len,1]
            x_weight=F.softmax(x_score.transpose(2,3),-1)#[B,tgt_vocab,1,seq_len]
            output=torch.matmul(x_weight,x_t).squeeze()#[B,tgt_vocab,1,dim]
        elif type=='add':
            x_t=x.transpose(1,2).contiguous()
            output=x_t.mean(dim=2,keepdim=False)
        elif type=='vocab_size_attn':
            pass

        else:
            output=None
        return output

class MultiAngleFusion(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.get_score_1=nn.Linear(dim,32)
        self.get_score_2=nn.Linear(32,1)
        # self.Multheadattn_trans_1_2=MultiHeadAttention(4,512,512,512)
        # self.Multheadattn_trans_2_1=MultiHeadAttention(4,512,512,512)
        # self.Multheadattn_trans_1_3=MultiHeadAttention(4,512,512,512)
        # self.Multheadattn_trans_3_1=MultiHeadAttention(4,512,512,512)
        # self.Multheadattn_trans_2_3=MultiHeadAttention(4,512,512,512)
        # self.Multheadattn_trans_3_2=MultiHeadAttention(4,512,512,512)
        # self.Multheadattn_fus_1=MultiHeadAttention(4,1024,1024,1024)
        # self.Multheadattn_fus_2=MultiHeadAttention(4,1024,1024,1024)
        # self.Multheadattn_fus_3=MultiHeadAttention(4,1024,1024,1024)
        # self.dense_align=nn.Linear(1024,512)

        self.get_angle_1=nn.Linear(dim,dim)
        self.get_angle_2=nn.Linear(dim,dim)
        self.get_angle_3=nn.Linear(dim,dim)
        self.out_put=nn.Linear(dim,dim)
  


    def forward(self,angle_1,angle_2,angle_3,type='add'):#[B,vocab_size,dim]
        if type=='add':
            output=(angle_1+angle_2+angle_3)/3
        elif type=='angleFus':
            angle_1=self.get_angle_1(angle_1)
            angle_2=self.get_angle_2(angle_2)
            angle_3=self.get_angle_3(angle_3)
            angle_all=torch.cat([angle_1.unsqueeze(2),angle_2.unsqueeze(2),angle_3.unsqueeze(2)],2)#[B,vovab_size,3,dim]
            angle_score=self.get_score_2(self.get_score_1(angle_all))
            angle_weight=F.softmax(angle_score.transpose(2,3))#[B,vocab_size,1,3]
            output=torch.matmul(angle_weight,angle_all).squeeze()
            output=self.out_put(output)
        elif type=='FusAdd':
            # stop()
            angle_1_2_cross,_=self.Multheadattn_trans_1_2(angle_1,angle_2,angle_2)
            angle_1_3_cross,_=self.Multheadattn_trans_1_3(angle_1,angle_3,angle_3)
            angle_1_cross=torch.cat([angle_1_2_cross,angle_1_3_cross],dim=-1)
            angle_1_cross_fus,_=self.Multheadattn_fus_1(angle_1_cross,angle_1_cross,angle_1_cross)

            angle_2_1_cross,_=self.Multheadattn_trans_2_1(angle_2,angle_1,angle_1)
            angle_2_3_cross,_=self.Multheadattn_trans_2_3(angle_2,angle_3,angle_3)
            angle_2_cross=torch.cat([angle_2_1_cross,angle_2_3_cross],dim=-1)
            angle_2_cross_fus,_=self.Multheadattn_fus_2(angle_2_cross,angle_2_cross,angle_2_cross)

            angle_3_1_cross,_=self.Multheadattn_trans_3_1(angle_3,angle_1,angle_1)
            angle_3_2_cross,_=self.Multheadattn_trans_3_2(angle_3,angle_2,angle_2)
            angle_3_cross=torch.cat([angle_3_1_cross,angle_3_2_cross],dim=-1)
            angle_3_cross_fus,_=self.Multheadattn_fus_3(angle_3_cross,angle_3_cross,angle_3_cross)
            
            output_mean=(angle_1_cross_fus+angle_2_cross_fus+angle_3_cross_fus)/3
            output=self.dense_align(output_mean)

        else:
            output=None
        return output



