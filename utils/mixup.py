import logging
import numpy as np
import torch
from torch.nn.functional import interpolate
class Mixup:
	def __init__(self, prob_apply=1., apply_per="elem", img_mix_func="mixup", alpha=.2):
		self.prob_apply = prob_apply
		self.apply_per = apply_per
		if (self.apply_per == "elem"):
			self.apply_fct = self._mixup_elem_param
		else:
			self.apply_fct = self._mixup_batch_param

		if(img_mix_func=="mixup"):
			self.img_mix_func = self.mixup
		elif(img_mix_func=="cutmix"):
			self.img_mix_func = self.cutmix
		elif(img_mix_func=="resize"):
			self.img_mix_func = self.resizeMix
		else:
			logging.info("ERROR mixup function {} not recognized".format(img_mix_func))

		self.alpha = alpha
		self.num_classes = None

	def forward(self, batch_x, batch_y):
		x1, y1, x2, y2 = self.sample_x1_x2(batch_x, batch_y)
		x, y, lam = self.apply_fct(x1, y1, x2, y2)
		return x, y, lam

	def sample_x1_x2(self, batch_x, batch_y):
		#y1 to onehot
		if (len(batch_y.shape) <= 1):  # not onehot targets, transform to onehot first
			assert self.num_classes is not None, "provide onehot target vectors or set the number of classes for onehot mixup target vectors"
			batch_y = torch.full((batch_y.size(0), self.num_classes), 0, device=batch_y.device).scatter_(1, batch_y.view(-1, 1), 1)

		bsz = batch_x.size(0)
		ind = torch.randperm(bsz)
		x2 = batch_x.clone()[ind, ...]
		y2 = batch_y.clone()[ind, ...]
		return batch_x, batch_y, x2, y2

	def _mixup_batch_param(self, x1, y1, x2, y2):
		"""
		linear combination of pairs of inputs and their labels (labels must be onehot vectors)
		samples 1 lambda for the entire batch of imgs
		"""
		if (np.random.rand() < self.prob_apply):
			# sample coeff lambda from beta distrib
			lam = torch.tensor(np.random.beta(self.alpha, self.alpha))
		else:
			lam = torch.tensor(1.)
		x, y = self.img_mix_func(x1, y1, x2, y2, lam)
		return x, y, lam

	def _mixup_elem_param(self, x1, y1, x2, y2):
		"""
		linear combination of pairs of inputs and their labels (labels must be onehot vectors)
		samples 1 lambda per image
		"""
		bsz = x1.size(0)
		lam = torch.tensor(np.random.beta(self.alpha, self.alpha, size=bsz), dtype=x1.dtype, requires_grad=False,
						   device=x1.device)
		apply_mask = np.random.rand(bsz) > self.prob_apply
		lam[apply_mask] = 1.
		x, y = self.img_mix_func(x1, y1, x2, y2, lam)
		return x,y,lam

	def mixup(self, x1, y1, x2, y2, lam):
		# generate x from mixup of x1 and x2
		x = lam.view(-1, 1, 1, 1) * x1 + (1. - lam.view(-1, 1, 1, 1)) * x2
		y = lam.view(-1, 1) * y1 + (1. - lam.view(-1, 1)) * y2
		return x, y

	def cutmix(self, x1, y1, x2, y2, lam):
		# x1 : [bsz, channels, width, height]
		bsz, C, W, H = x1.size()

		# width and height of cutout area
		r_w = np.floor(np.sqrt(1 - lam.cpu().numpy()) * W)
		r_h = np.floor(np.sqrt(1 - lam.cpu().numpy()) * H)

		# starting pos of the cutout area in W and H axes
		cx  = torch.randint(W, size=(bsz,), requires_grad=False)
		cy = torch.randint(H, size=(bsz,), requires_grad=False)
		
		p_w1 = torch.clip(cx - r_w//2, 0, W).to(x1.device)
		p_h1 = torch.clip(cy - r_h//2, 0, H).to(x1.device)
		p_w2 = torch.clip(cx + r_w//2, 0, W).to(x1.device)
		p_h2 = torch.clip(cy + r_h//2, 0, H).to(x1.device)

		# real value of lambda due to rounding of pixel positions
		lam = 1 - ((p_w2 - p_w1) * (p_h2 - p_h1) / (W * H))
		x = self._box_mix_tensors(x1, x2, p_w1, p_w2, p_h1, p_h2)
		y = lam.view(-1, 1) * y1 + (1. - lam.view(-1, 1)) * y2
		return x,y

	def resizeMix(self, x1, y1, x2, y2, lam):
		# x1 : [bsz, channels, width, height]
		bsz, C, W, H = x1.size()

		# width and height of cutout area
		r_w = np.floor(np.sqrt(1 - lam.cpu().numpy()) * W)
		r_h = np.floor(np.sqrt(1 - lam.cpu().numpy()) * H)

		# starting pos of the cutout area in W and H axes
		cx  = torch.randint(W, size=(bsz,), requires_grad=False)
		cy = torch.randint(H, size=(bsz,), requires_grad=False)
		
		p_w1 = torch.clip(cx - r_w//2, 0, W).to(x1.device)
		p_h1 = torch.clip(cy - r_h//2, 0, H).to(x1.device)
		p_w2 = torch.clip(cx + r_w//2, 0, W).to(x1.device)
		p_h2 = torch.clip(cy + r_h//2, 0, H).to(x1.device)
		imgs_resize = interpolate(x2, (p_h2 - p_h1, p_w2 - p_w1), mode="nearest")

		# real value of lambda due to rounding of pixel positions
		lam = 1 - ((p_w2 - p_w1) * (p_h2 - p_h1) / (W * H))
		x = self._box_mix_tensors(x1, imgs_resize, p_w1, p_w2, p_h1, p_h2)
		y = lam.view(-1, 1) * y1 + (1. - lam.view(-1, 1)) * y2
		return x,y

	def _box_mix_tensors(self, tensor1, tensor2, x1, x2, y1, y2):
		"""
			used for cutmix.
			create a tensor of shape [bsz, C, W, H] where values outside of the [W, H] boxes [x1:x2, y1:y2] are from tensor1
			and values inside the box are from tensor2.
			x1, x2, y1, y2 are expected to be values of size [bsz]
		"""
		bsz, C, W, H = tensor1.size()
		w = torch.arange(W).expand(W, H)
		h = torch.arange(H).expand(W, H).T
		inds_w = w.expand(bsz, C, W, H).to(tensor1.device)
		inds_h = h.expand(bsz, C, W, H).to(tensor1.device)
		m_w = torch.logical_and(inds_w >=x1.view(bsz, 1, 1, 1), inds_w < x2.view(bsz, 1, 1, 1))
		m_h = torch.logical_and(inds_h >=y1.view(bsz, 1, 1, 1), inds_h < y2.view(bsz, 1, 1, 1))
		m = torch.logical_and(m_w, m_h)#mask 1 if inside boxes, 0 elsewhere
		tensor_mixed = torch.where(m, tensor2, tensor1)
		return tensor_mixed

	def setup_mixup_istep(self, iStep, nbr_classes):
		self.num_classes = nbr_classes
		logging.info("mixup set up for iStep {} with {} classes.".format(iStep, nbr_classes))

class RehearsalMixup(Mixup):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def setup_mixup_istep(self, iStep, nbr_classes, infiniteMemoryLoader):
		self.num_classes = nbr_classes
		self.x2_loader = infiniteMemoryLoader
		logging.info("memory loader set for mixup rehearsal")
		logging.info("mixup set up for iStep {} with {} classes.".format(iStep, nbr_classes))
	def sample_x1_x2(self, batch_x, batch_y):
		#y1 to onehot
		if (len(batch_y.shape) <= 1):  # not onehot targets, transform to onehot first
			assert self.num_classes is not None, "provide onehot target vectors or set the number of classes for onehot mixup target vectors"
			batch_y = torch.full((batch_y.size(0), self.num_classes), 0, device=batch_y.device).scatter_(1, batch_y.view(-1, 1), 1)

		assert self.x2_loader is not None, "provide x2 sampling loader for mixup if not using randperm"
		bsz = batch_x.size(0)
		_, x2, y2 = self.x2_loader.next()
		x2 = x2[:bsz,...].to(batch_x.device)
		y2 = y2[:bsz].to(batch_y.device)
		if (len(y2.shape) <= 1):  # not onehot targets, transform to onehot first
			assert self.num_classes is not None, "provide onehot target vectors or set the number of classes for onehot mixup target vectors"
			y2 = torch.full((y2.size(0), self.num_classes), 0, dtype=torch.float32, device=batch_y.device).scatter_(1, y2.view(-1, 1), 1)
		return batch_x, batch_y, x2, y2