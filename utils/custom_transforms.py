import numpy as np
import random
import torchvision.transforms.functional as TF
import torch
class N_augmentedViewsTransform:
	r"""
		Transform used for contrastive learning, returns n views of the same image.
		The first view is the base transformed image, and the other views are obtained with contrastive transforms
	"""
	def __init__(self, transform, n, contrastiveTransform=None):
		self.transform = transform
		self.contrastiveTransform = contrastiveTransform if contrastiveTransform is not None else transform
		self.n = n

	def __call__(self, x):
		#return list= [tensor_image, tensor_views]
		image = self.transform(x)
		views = [self.contrastiveTransform(x) for i in range(1,self.n)]
		return [image]+views

	def __str__(self):
		return "N_augmentedViewsTransform(\n    n="+str(self.n)+"\n    transform="+str(self.transform)+"\n    contrastiveTransform="+str(self.contrastiveTransform)+"\n)"

class RandomRotationDiscrete:
	r"""
		mimics torchvision.transforms.RandomRotation but chooses angles from a discrete list of angles
		instead of a continuous interval
	"""
	def __init__(self, angles):
		self.angles = angles#angles must be a list of angles to choose from

	def __call__(self, x):
		angle = random.choice(self.angles)
		return TF.rotate(x, angle)
	def __str__(self):
		return "RandomRotationDiscrete(angles="+str(self.angles)+")"

class Cutout(object):
	def __init__(self, n_holes, length):
		self.n_holes = n_holes
		self.length = length

	def __call__(self, img):
		h = img.size(1)
		w = img.size(2)

		mask = np.ones((h, w), np.float32)

		for n in range(self.n_holes):
			y = np.random.randint(h)
			x = np.random.randint(w)

			y1 = np.clip(y - self.length // 2, 0, h)
			y2 = np.clip(y + self.length // 2, 0, h)
			x1 = np.clip(x - self.length // 2, 0, w)
			x2 = np.clip(x + self.length // 2, 0, w)

			mask[y1: y2, x1: x2] = 0.

		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img = img * mask

		return img
	def __str__(self):
		return "Cutout(n_holes="+str(self.n_holes)+", length="+str(self.length)+")"