import argparse,math,time,warnings,copy, numpy as np, os.path as path 
import utils.evals as evals
import utils.utils as utils
from utils.data_loader import process_data
import torch, torch.nn as nn, torch.nn.functional as F
import lamp.Constants as Constants
from lamp.Models import LAMP
from lamp.Translator import translate
from config_args import config_args,get_args
from pdb import set_trace as stop
from tqdm import tqdm




def train_epoch(model,train_data, crit, optimizer,adv_optimizer,epoch,data_dict,opt):
	model.train()
	out_len = (opt.tgt_vocab_size) if opt.binary_relevance else (opt.tgt_vocab_size-1)

	all_predictions = torch.zeros(len(train_data._src_text_insts),out_len)
	all_targets = torch.zeros(len(train_data._src_text_insts),out_len)

	
	batch_idx,batch_size,n_batches = 0,train_data._batch_size,train_data._n_batch
	bce_total,d_total,d_fake_total,g_total = 0,0,0,0
	src_len=len(train_data._src_text_insts)
	# stop()
	#multi-batch
	accum_count=opt.accum_count
	True_batches=[]
	batch_acc=0
	# B_batches=src_len//(batch_size*accum_count) # the num of the accum batches
	B_batches=0 # the num of the accum batches
	bce_total=0
	# stop()
	for batch in tqdm(train_data, mininterval=0.5,desc='(Training)', leave=False):
		True_batches.append(batch)
		batch_acc+=1
		batch_idx+=1
		# loss,pre_out,tgt_out=train_batch(model,crit,batch,opt)

		# optimizer.step()

		if batch_acc==accum_count or batch_idx==n_batches:
			# stop()
			loss_item,pred_out,tgt_out=train_all_batches(model,crit,optimizer,opt,epoch,True_batches)
			# if batch_idx==n_batches:
			# 	import pdb; pdb.set_trace()
			## Updates ##
			start_idx, end_idx = (B_batches*batch_size),((B_batches+accum_count)*batch_size)
			all_predictions[start_idx:end_idx] = pred_out
			all_targets[start_idx:end_idx] = tgt_out
			B_batches +=batch_acc
			optimizer.step()
			bce_total+=loss_item
			# B_batches-=1
			batch_acc=0
			True_batches.clear()
		
	
	return all_predictions, all_targets, bce_total
def train_all_batches(model,crit,optimizer,opt,epoch,batches):
	pred_out_all=[]
	tgt_out_all=[]
	loss_all=0
	optimizer.zero_grad()
	for batch in batches:
		loss,tgt_out,pred_out=train_batch(model,crit,opt,epoch,batch)
		loss.backward()
		loss_all+=loss.item()
		pred_out_all.append(pred_out)
		tgt_out_all.append(tgt_out)
	pred=torch.cat(pred_out_all,0)
	tgt=torch.cat(tgt_out_all,0)
	return loss_all,pred,tgt

	
def train_batch(model,crit,opt,epoch,batch):	

	src,adj,tgt = batch
	loss,d_loss = 0,0
	gold = tgt[:, 1:]

	if opt.binary_relevance:
		gold_binary = utils.get_gold_binary(gold.data.cpu(),opt.tgt_vocab_size).cuda()
		# optimizer.zero_grad()
		# stop()
		pred,enc_output,*results = model(src,adj,None,gold_binary,return_attns=opt.attns_loss,int_preds=opt.int_preds)
		norm_pred = F.sigmoid(pred)
		bce_loss =  F.binary_cross_entropy_with_logits(pred, gold_binary,reduction='mean')
		loss += bce_loss
		# bce_total += bce_loss.item()
		if opt.int_preds and not opt.matching_mlp:
			for i in range(len(results[0])):
				bce_loss =  F.binary_cross_entropy_with_logits(results[0][i], gold_binary,reduction='mean')
				loss += (opt.int_pred_weight)*bce_loss
		# if epoch == opt.thresh1:
		# 	opt.init_model = copy.deepcopy(model)
		# loss.backward()
		# optimizer.step()
		tgt_out = gold_binary.data
		pred_out = norm_pred.data
		return loss,tgt_out,pred_out

	else: 
		# Non Binary Outputs
		# optimizer.zero_grad()
		pred,enc_output,*results = model(src,adj,tgt,None,int_preds=opt.int_preds)
		loss = crit(F.log_softmax(pred), gold.contiguous().view(-1))
		pred = F.softmax(pred,dim=1)
		pred_vals,pred_idxs = pred.max(1)
		pred_vals = pred_vals.view(gold.size()).data.cpu()
		pred_idxs = pred_idxs.view(gold.size()).data.cpu()
		pred_out = torch.zeros(pred_vals.size(0),pred.size(1)).scatter_(1,pred_idxs.long(),pred_vals)
		tgt_out = torch.zeros(pred_vals.size(0),pred.size(1)).scatter_(1,gold.data.cpu().long(),torch.ones(pred_vals.size()))
		pred_out = pred_out[:,1:]
		tgt_out = tgt_out[:,1:]
		# loss.backward()
		# optimizer.step()
		return loss,pred_out,tgt_out