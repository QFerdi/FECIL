from torch.optim.lr_scheduler import _LRScheduler

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.call_at_end = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                    self.call_at_end = self.after_scheduler.call_at_end
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.)
                for base_lr in self.base_lrs
            ]
    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

class WarmupScheduler(_LRScheduler):
    """
    Gradual Warmup scheduler implementation to be called at the end of epochs
    """
    def __init__(self, optimizer, warmup_epochs, after_scheduler, **kwargs):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super(WarmupScheduler, self).__init__(optimizer, **kwargs)
        #self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        self.last_epoch = -1
        #init lr for 1st epoch
        for i,group in enumerate(optimizer.param_groups):
                group['lr'] = self.base_lrs[i]*(1/self.warmup_epochs)#init 1/epochs * lr
    def get_lr(self):
        if(self.finished):
            return self.after_scheduler.get_lr()
        else:
            if(self.last_epoch+2>=self.warmup_epochs and not self.finished):
                self.finished = True
            return [base_lr * (float(self.last_epoch+2) / self.warmup_epochs) for base_lr in self.base_lrs]
        pass
    def step(self):#at the end of epoch
        if(self.finished):
            self.after_scheduler.step()
        else:
            super().step()