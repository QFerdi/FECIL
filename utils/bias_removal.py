import logging
from tqdm import tqdm
import numpy as np
import torch
from torch.nn import functional as F
from utils.toolkit import tensor2numpy

def WA(fc_layer, nb_past_classes, nb_new_classes):
	all_seen_classes = nb_past_classes+nb_new_classes
	weights=fc_layer.weight.data
	newnorm=(torch.norm(weights[nb_past_classes:all_seen_classes,:],p=2,dim=1))
	oldnorm=(torch.norm(weights[:nb_past_classes,:],p=2,dim=1))
	meannew=torch.mean(newnorm)
	meanold=torch.mean(oldnorm)
	gamma=meanold/meannew
	logging.info('weight alignment, gamma={}'.format(gamma))
	if(gamma < 1.):#weights are biased, align new norm
		fc_layer.weight.data[nb_past_classes:all_seen_classes,:] *= gamma
	else:
		logging.info("no new classes\' bias detected, no alignment")
	return gamma

def WA_v2(fc_layer, nb_past_classes, nb_new_classes):
	all_seen_classes = nb_past_classes+nb_new_classes
	weights=fc_layer.weight.data
	if(fc_layer.bias is not None):
		b = fc_layer.bias.data
		weights = torch.cat((weights, b.reshape(-1, 1)), dim=1)
	newnorm=(torch.norm(weights[nb_past_classes:all_seen_classes,:],p=2,dim=1))
	oldnorm=(torch.norm(weights[:nb_past_classes,:],p=2,dim=1))
	meannew=torch.mean(newnorm)
	meanold=torch.mean(oldnorm)
	gamma=meanold/meannew
	logging.info('weight alignment, gamma={}'.format(gamma))
	if(gamma < 1.):#weights are biased, align new norm
		fc_layer.weight.data[nb_past_classes:all_seen_classes,:] *= gamma
		if(fc_layer.bias is not None):
			fc_layer.bias.data[nb_past_classes:all_seen_classes] *= gamma
	else:
		logging.info("no new classes\' bias detected, no alignment")
	return gamma

def WA_v3(fc_layer, nb_past_classes, nb_new_classes):
	all_seen_classes = nb_past_classes+nb_new_classes
	weights=fc_layer.weight.data
	newnorm=(torch.norm(weights[nb_past_classes:all_seen_classes,:],p=2,dim=1))
	oldnorm=(torch.norm(weights[:nb_past_classes,:],p=2,dim=1))
	meannew=torch.mean(newnorm)
	meanold=torch.mean(oldnorm)
	gamma=meanold/meannew
	logging.info('weight alignment, gamma={}'.format(gamma))
	if(gamma < 1.):#weights are biased, align new norm
		fc_layer.weight.data[nb_past_classes:all_seen_classes,:] *= gamma
		if(fc_layer.bias is not None):
			b = fc_layer.bias.data
			gamma_b = torch.mean(torch.abs(b[:nb_past_classes]))/torch.mean(torch.abs(b[nb_past_classes:all_seen_classes]))
			logging.info('weight alignment, bias gamma={}'.format(gamma))
			if(gamma_b < 1.):
				fc_layer.bias.data[nb_past_classes:all_seen_classes,:] *= gamma_b
	else:
		logging.info("no new classes\' bias detected, no alignment")
	return gamma

def compute_bias(fc_layer, nb_newClasses):
	if(type(fc_layer).__name__ == "SplitCosineLinear"):
		weights_new = fc_layer.fc2.weight.data
		weights_old = fc_layer.fc1.weight.data
		newnorm = torch.norm(weights_new,p=2,dim=1)
		oldnorm = torch.norm(weights_old,p=2,dim=1)
	else:
		weights = fc_layer.weight.data
		newnorm=(torch.norm(weights[-nb_newClasses:,:],p=2,dim=1))
		oldnorm=(torch.norm(weights[:-nb_newClasses,:],p=2,dim=1))

	meannew=torch.mean(newnorm)
	meanold=torch.mean(oldnorm)
	gamma=meanold/meannew
	return gamma

def balanced_finetuning(balanced_loader,
                      test_loader,
                      optimizer,
                      model,
                      nbrEpochs,
                      eval_func,
                      output_keyword="logits",
                      epochs_between_evals=1,
                      metricsLog=None,
                      device=torch.device('cpu')):

	prog_bar = tqdm(range(nbrEpochs))
	for _, epoch in enumerate(prog_bar):
		model.train()
		correct, total = 0, 0
		for i, (_, inputs, targets) in enumerate(balanced_loader):
			inputs, targets = inputs.to(device), targets.to(device)
			logits = model(inputs)[output_keyword]
			loss = F.cross_entropy(logits,targets)
			if(metricsLog is not None):
				metricsLog.info("train_loss", loss_tot=loss)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			_, preds = torch.max(logits, dim=1)
			correct += preds.eq(targets.expand_as(preds)).cpu().sum()
			total += len(targets)

		train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
		if epoch%epochs_between_evals == 0:
			#end-of-epoch eval
			epoch_eval = eval_func(model, test_loader)
			metricsLog.info("epoch_eval", **epoch_eval)
		metricsLog.info("epoch_end", prog_bar, train_acc)