import logging
import numpy as np
from tqdm import tqdm
import torch
import copy
from torch import nn
from torch import optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from learners.base import BaseLearner
from utils.inc_net import DualFeatureNet, IncrementalNet
from losses.base import KD_loss, KLdiv_loss, CEsoft_loss
from utils.mixup import RehearsalMixup
from utils.data_manager import InfiniteLoader
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.metrics_factory import topk_accuracy, newOld_accuracy
from utils.schedulers import WarmupScheduler
from utils.bias_removal import WA, compute_bias
class FECIL(BaseLearner):
	def __init__(self, args):
		super().__init__(args)
		self.args = args
		self._network = IncrementalNet(args.convnet_type, False)
		self._new_net = DualFeatureNet(args.convnet_type, False, use_auxfc=True)
		self.metricsLog = args.metricsLogger #metrics computation
		self.epochs_between_evals = 1
		self.uncompressed_accs = []

		#init mixup
		self.rehearsal_mixup = RehearsalMixup(alpha=self.args.alpha_mixup, prob_apply=1., img_mix_func=self.args.mix_func)

	def after_task(self):
		self._known_classes = self._total_classes
		logging.info('Exemplar size: {}'.format(self.exemplar_size))

	def incremental_train(self, data_manager):
		self._cur_task += 1
		self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
		logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

		#get loaders
		self.setup_data(data_manager)

		if(self._cur_task==0):
			#add outputs for new classes to compressed model
			self._network.update_fc(self._total_classes)
			self._network.to(self._device)
			if len(self._multiple_gpus) > 1:
				self._network = nn.DataParallel(self._network, self._multiple_gpus)
			self._train_init(self.expansion_loader, self.test_loader)
		else:
			#init both F_extractors to prev compressed_net and add outputs to fc for new classes
			self._new_net.update_model(self._network, self._total_classes)
			#freeze prev feature extractor
			self._new_net.partial_freeze()
			self._new_net.to(self._device)
			if len(self._multiple_gpus) > 1:
				self._new_net = nn.DataParallel(self._new_net, self._multiple_gpus)
			self.expansion_phase(self.expansion_loader, self.test_loader)
			#classifier bias correction
			if(self.args.H_big_bias_correction == "WA"):#weight alignment
				self.align_net(self._new_net)
			else:
				logging.info("no weight alignment after expansion !")
				bias_gamma = compute_bias(self._new_net.fc, self._total_classes-self._known_classes)
				logging.info("expanded model bias : {}".format(bias_gamma))

			#eval dynamic model
			eval_bigModel = self._epoch_eval(self._new_net, self.test_loader)
			acc = topk_accuracy(eval_bigModel['preds_cnn'], eval_bigModel['labels'])
			accOld, accNew = newOld_accuracy(eval_bigModel['preds_cnn'], eval_bigModel['labels'], self._known_classes)
			logging.info("Final accuracy of big Model {}% ( {}% Old, {}% new)".format(acc, accOld, accNew))
			self.uncompressed_accs.append(acc)
			logging.info("Big Model acc curve {}".format(self.uncompressed_accs))

			#add outputs for new classes to compressed model
			self._network.update_fc(self._total_classes)
			#freeze big model
			self._network.to(self._device)
			if len(self._multiple_gpus) > 1:
				self._new_net.module.freeze()
				self._network = nn.DataParallel(self._network, self._multiple_gpus)
			else:
				self._new_net.freeze()
			self.compression_phase(self.compression_loader, self.test_loader)

			#classifier bias correction
			if(self.args.H_bias_correction == "WA"):#weight alignment
					self.align_net(self._network)
			else:
				logging.info("no weight alignment after compression !")
				if(len(self._multiple_gpus)>1):
					fc = self._network.module.fc
				else:
					fc = self._network.fc
				bias_gamma = compute_bias(fc, self._total_classes-self._known_classes)
				logging.info("compressed model bias : {}".format(bias_gamma))
			if(len(self._multiple_gpus)>1):
				self.metricsLog.info("classifier_calibration", self._network.module.fc)
			else:
				self.metricsLog.info("classifier_calibration", self._network.fc)

			eval_compression = self._epoch_eval(self._network, self.test_loader)
			acc = topk_accuracy(eval_compression['preds_cnn'], eval_compression['labels'])
			accOld, accNew = newOld_accuracy(eval_compression['preds_cnn'], eval_compression['labels'], self._known_classes)
			logging.info("Final accuracy of Compressed Model {}% ( {}% Old, {}% new)".format(acc, accOld, accNew))

		#update rehearsal memory
		self.build_rehearsal_memory(data_manager, self.samples_per_class)
		#reset DataParallel for next step
		if len(self._multiple_gpus) > 1:
			if(isinstance(self._network, nn.DataParallel)):
				self._network = self._network.module
			if(isinstance(self._new_net, nn.DataParallel)):
				self._new_net = self._new_net.module

	def get_loader(self, dataset, shuffle=True, drop_last=False):
		#drop_last true for faster epochs if
		#torch.backends.cudnn.benchmark = True in trainer.py
		loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=shuffle,
							num_workers=self.args.num_workers, pin_memory=True, drop_last=drop_last)
		return loader

	def setup_data(self, data_manager):
		#get incremental dataset
		train_expansion_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
															source='train', mode='train', appendent=self._get_memory())
		train_compression_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
																source='train', mode='train',
																appendent=self._get_memory())
		test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
		self.expansion_loader = self.get_loader(train_expansion_dataset)
		self.compression_loader = self.get_loader(train_compression_dataset)
		self.test_loader = self.get_loader(test_dataset, shuffle=False)

		logging.info('Expansion set size : {}, Compression set size : {}, test set size : {}'.format(len(train_expansion_dataset), len(train_compression_dataset), len(test_dataset)))
		logging.info('Expansion transforms : {}'.format(train_expansion_dataset.trsf))
		logging.info('Compression transforms : {}'.format(train_compression_dataset.trsf))


		#get mixup rehearsal dataset (ie. infinite loader of memory dataset)
		if(self._cur_task>0):
			train_mixup_set = data_manager.get_dataset([], source='train', mode='train', appendent=self._get_memory())
			train_mixup_loader = InfiniteLoader(self.get_loader(train_mixup_set, shuffle=True, drop_last=True))
			self.rehearsal_mixup.setup_mixup_istep(self._cur_task, self._total_classes, train_mixup_loader)
			logging.info('R-mixup transforms : {}'.format(train_mixup_set.trsf))

	def get_optim_init(self):
		init_epochs = self.args.init_epochs if not self.args.warmup else self.args.init_epochs - self.args.warmup_epochs
		optimizer = optim.SGD(self._network.parameters(), momentum=0.9,
								  lr=self.args.init_lr,weight_decay=self.args.init_weight_decay)
		if(self.args.cosine_scheduler):
			scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=init_epochs)
		else:
			scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.init_milestones,
														   gamma=self.args.init_lr_decay)
		if(self.args.warmup):
			scheduler = WarmupScheduler(optimizer=optimizer, warmup_epochs=self.args.warmup_epochs, after_scheduler=scheduler)
		return optimizer, scheduler

	def get_optim_expansion(self):
		exp_epochs = self.args.exp_epochs if not self.args.warmup else self.args.exp_epochs-self.args.warmup_epochs
		optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._new_net.parameters()), momentum=0.9,
								  lr=self.args.lrate, weight_decay=self.args.weight_decay)
		if(self.args.cosine_scheduler):
			scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=exp_epochs)
		else:
			scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.milestones,
														   gamma=self.args.lr_decay)
		if(self.args.warmup):
			scheduler = WarmupScheduler(optimizer=optimizer, warmup_epochs=self.args.warmup_epochs, after_scheduler=scheduler)
		return optimizer, scheduler

	def get_optim_compression(self):
		compress_epochs = self.args.compress_epochs if not self.args.warmup else self.args.compress_epochs - self.args.warmup_epochs
		optimizer = optim.SGD(self._network.parameters(), momentum=0.9,
								  lr=self.args.lrate, weight_decay=self.args.weight_decay)
		if(self.args.cosine_scheduler):
			scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=compress_epochs)
		else:
			scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args.milestones,
														   gamma=self.args.lr_decay)
		if(self.args.warmup):
			scheduler = WarmupScheduler(optimizer=optimizer, warmup_epochs=self.args.warmup_epochs, after_scheduler=scheduler)
		return optimizer, scheduler

	def _train_init(self, train_loader, test_loader):
		logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
		#get optim
		optimizer, scheduler = self.get_optim_init()
		#train
		logging.info("nb_batches per epoch : {}".format(len(train_loader)))
		prog_bar = tqdm(range(self.args.init_epochs))
		loss_log = {}
		for _, epoch in enumerate(prog_bar):
			self._network.train()
			correct, total = 0, 0
			for i, (_, inputs, targets) in enumerate(train_loader):
				inputs, targets = inputs.to(self._device), targets.to(self._device)
				outputs = self._network(inputs)
				logits = outputs['logits']
				loss=F.cross_entropy(logits,targets)

				loss_log["loss_tot"] = loss

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				#log losses
				self.metricsLog.info("train_loss", **loss_log)

				_, preds = torch.max(logits, dim=1)
				correct += preds.eq(targets.expand_as(preds)).cpu().sum()
				total += len(targets)

			scheduler.step()
			train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)

			if epoch%self.epochs_between_evals==0:
				epoch_eval = self._epoch_eval(self._network, test_loader)
				self.metricsLog.info("epoch_eval", **epoch_eval)
			self.metricsLog.info("epoch_end", prog_bar, train_acc)

	def expansion_phase(self, train_loader, test_loader):
		logging.info('Trainable params: {}'.format(count_parameters(self._new_net, True)))
		#get optim
		optimizer, scheduler = self.get_optim_expansion()
		#train
		logging.info("nb_batches per epoch : {}".format(len(train_loader)))
		prog_bar = tqdm(range(self.args.exp_epochs))
		if(self.args.expansion_mixup):
			Lce = CEsoft_loss
		else:
			Lce = F.cross_entropy
		loss_log = {}
		for _, epoch in enumerate(prog_bar):
			if len(self._multiple_gpus) > 1:
				self._new_net.module.train()
				self._new_net.module.feat_old.eval()
			else:
				self._new_net.train()
				self._new_net.feat_old.eval()

			correct, total = 0, 0
			for i, (_, inputs, targets) in enumerate(train_loader):
				inputs, targets = inputs.to(self._device), targets.to(self._device)
				if(self.args.expansion_mixup):
					inputs,targets,lam = self.rehearsal_mixup.forward(inputs, targets)
					aux_targets = targets.clone()
					aux_targets = torch.cat((torch.sum(aux_targets[:, :self._known_classes], dim=1, keepdim=True),
						                        aux_targets[:,self._known_classes:]), dim=1)
				else:
					aux_targets = targets.clone()
					aux_targets=torch.where(aux_targets-self._known_classes+1>0,aux_targets-self._known_classes+1,0)

				outputs = self._new_net(inputs)
				logits=outputs["logits"]
				loss_clf = Lce(logits, targets)
				loss = loss_clf
				
				aux_logits = outputs["aux_logits"]
				loss_clfNew=Lce(aux_logits,aux_targets)
				loss += loss_clfNew

				loss_log.update(loss_aux=loss_clfNew)
				loss_log.update(loss_clf=loss, loss_tot=loss)

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				#log losses
				self.metricsLog.info("train_loss", **loss_log)

				_, preds = torch.max(logits, dim=1)
				correct += preds.eq(targets.expand_as(preds)).cpu().sum()
				total += len(targets)

			scheduler.step()
			train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
			if epoch%self.epochs_between_evals==0:
				#end-of-epoch eval
				epoch_eval = self._epoch_eval(self._new_net, test_loader)
				self.metricsLog.info("epoch_eval", **epoch_eval)
			self.metricsLog.info("epoch_end", prog_bar, train_acc)
	def compression_phase(self, train_loader, test_loader):
		logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
		#get optim
		optimizer, scheduler = self.get_optim_compression()
		#train
		logging.info("starting compression")
		logging.info("nb_batches per epoch : {}".format(len(train_loader)))
		prog_bar = tqdm(range(self.args.compress_epochs))
		self._new_net.eval()
		for _, epoch in enumerate(prog_bar):
			self._network.train()
			correct, total = 0, 0
			for i, (_, inputs, targets) in enumerate(train_loader):
				inputs, targets = inputs.to(self._device), targets.to(self._device)
				onehotTargets = target2onehot(targets, self._total_classes)
				if(self.rehearsal_mixup is not None):
					inputs,y, _ = self.rehearsal_mixup.forward(inputs, onehotTargets)
					targets = torch.max(y, dim=1)[1]

				out = self._network(inputs)
				logits=out["logits"]
				with torch.no_grad():
					teacher_out = self._new_net(inputs)
					teacher_logits = teacher_out["logits"]

				loss_kd=self.args.T**2*KLdiv_loss(logits,teacher_logits, self.args.T)

				loss_log = {"loss_kd":loss_kd}
				loss = loss_kd

				loss_log.update({"loss_compress":loss, "loss_tot":loss})

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				#log losses
				self.metricsLog.info("train_loss", **loss_log)

				_, preds = torch.max(logits, dim=1)
				correct += preds.eq(targets.expand_as(preds)).cpu().sum()
				total += len(targets)

			scheduler.step()
			train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
			if epoch%self.epochs_between_evals == 0:
				epoch_eval = self._epoch_eval(self._network, test_loader)
				self.metricsLog.info("epoch_eval", **epoch_eval)
			self.metricsLog.info("epoch_end", prog_bar, train_acc)

	@property
	def samples_old_class(self):
		if self._fixed_memory:
			return self._memory_per_class
		else:
			assert self._total_classes != 0, 'Total classes is 0'
			return (self._memory_size // self._known_classes)
	def samples_new_class(self, index):
		if self.args.dataset == "cifar100":
			return 500
		else:
			return self.data_manager.getlen(index)
	def align_net(self, net):
		#weight alignment
		if len(self._multiple_gpus) > 1:
			WA(net.module.fc, self._known_classes, self._total_classes-self._known_classes)
		else:
			WA(net.fc, self._known_classes, self._total_classes-self._known_classes)

	def _extract_vectors_dynamicModel(self, loader):
		self._new_net.eval()
		vectors, targets = [], []
		for _, _inputs, _targets in loader:
			_targets = _targets.numpy()
			if isinstance(self._new_net, nn.DataParallel):
				_vectors = tensor2numpy(self._new_net.module.extract_vector(_inputs.to(self._device)))
			else:
				_vectors = tensor2numpy(self._new_net.extract_vector(_inputs.to(self._device)))

			vectors.append(_vectors)
			targets.append(_targets)

		return np.concatenate(vectors), np.concatenate(targets)