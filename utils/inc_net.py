import logging
import itertools
import copy
import random
import torch
from torch import nn
from torch.nn import functional as F
from models.cifar_resnet import resnet32
from models.resnet import resnet18, resnet34, resnet50
from models.ucir_cifar_resnet import resnet32 as cosine_resnet32
from models.ucir_resnet import resnet18 as cosine_resnet18
from models.ucir_resnet import resnet34 as cosine_resnet34
from models.ucir_resnet import resnet50 as cosine_resnet50
from models.linears import SimpleLinear, SplitCosineLinear, CosineLinear


def get_convnet(convnet_type, pretrained=False):
    name = convnet_type.lower()
    if name == 'resnet32':
        return resnet32()
    elif name == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif name == 'cosine_resnet18':
        return cosine_resnet18(pretrained=pretrained)
    elif name == 'cosine_resnet32':
        return cosine_resnet32()
    elif name == 'cosine_resnet34':
        return cosine_resnet34(pretrained=pretrained)
    elif name == 'cosine_resnet50':
        return cosine_resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError('Unknown type {}'.format(convnet_type))


class BaseNet(nn.Module):

    def __init__(self, convnet_type, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet(convnet_type, pretrained)
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)['features']

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        '''
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        '''
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class IncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False):
        super().__init__(convnet_type, pretrained)
        self.gradcam = gradcam
        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.weight.data[:nb_output] = weight
            if(fc.bias is not None):
                bias = copy.deepcopy(self.fc.bias.data)
                fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc
    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        if(gamma < 1.):#weights are biased
            logging.info('alignweights,gamma={}'.format(gamma))
            self.fc.weight.data[-increment:,:]*=gamma
        return gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        out.update(x)
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out

    def forwardMixup(self, x, data_mixup = None, x2_mixup = None, layer_mix=-1, use_prev_mm_params=False):
        if(data_mixup is not None):
            if(layer_mix == -1):
                layer_mix = 4#random.randint(0,4)
            if(not use_prev_mm_params):
                data_mixup.set_manifold_mixup_params(x)
        x = self.convnet(x, data_mixup, x2_mixup, layer_mix)
        out = self.fc(x['features'])
        out.update(x)
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class ContrastiveINet(IncrementalNet):
    def __init__(self, *args, head='mlp', contrastive_dims=128, **kwargs):
        super().__init__(*args, **kwargs)
        if head == 'linear':
            self.head = nn.Linear(self.convnet.out_dim, contrastive_dims)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.convnet.out_dim, self.convnet.out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.convnet.out_dim, contrastive_dims)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        x = self.convnet(x)
        feats_contrast = F.normalize(self.head(x['features']), dim=1)
        out = self.fc(x['features'])
        out.update(x)
        out.update({'feats_contrast' : feats_contrast})
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations

        return out


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, convnet_type, pretrained, bias_correction=False):
        super().__init__(convnet_type, pretrained)

        # Bias layer
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        #self.bias_layer = BiasLayer()
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x['features'])
        if self.bias_correction:
            logits = out['logits']
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i+1]))
            #logits = self.bias_layer(logits, sum(self.task_sizes[:-1]), sum(self.task_sizes))
            out['logits'] = logits

        out.update(x)

        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())
        #params = self.bias_layer.get_params()

        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DERNet(nn.Module):
    def __init__(self, convnet_type, pretrained):
        super(DERNet,self).__init__()
        self.convnet_type=convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained=pretrained
        self.out_dim=None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features
    def forward(self, x):
        features = [convnet(x)['features'] for convnet in self.convnets]
        features = torch.cat(features, 1)

        out=self.fc(features) #{logics: self.fc(features)}

        aux_logits=self.aux_fc(features[:,-self.out_dim:])["logits"]

        out.update({"aux_logits":aux_logits,"features":features})
        '''{
            'features': features
            'logits': logits
            'aux_logits':aux_logits
        }'''
        return out

    def update_fc(self, nb_classes):
        if len(self.convnets)==0:
            self.convnets.append(get_convnet(self.convnet_type))
        else:
            self.convnets.append(get_convnet(self.convnet_type))
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

        self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()
    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        if(gamma < 1.):#weights are biased
            logging.info('alignweights,gamma={}'.format(gamma))
            self.fc.weight.data[-increment:,:]*=gamma
        return gamma


class DualFeatureNet(nn.Module):
    def __init__(self, convnet_type, pretrained, use_auxfc=True):
        super(DualFeatureNet,self).__init__()
        self.convnet_type=convnet_type
        self.pretrained=pretrained
        self.feat_old = get_convnet(self.convnet_type)
        self.feat_new = get_convnet(self.convnet_type)
        self.out_dim=self.feat_old.out_dim
        self.fc = None
        self.aux_fc=None
        self.old_fc = None
        self.use_auxfc = use_auxfc
        self.expansion_mode = True

    @property
    def feature_dim(self):
        return self.out_dim*2

    def extract_vector(self, x):
        features = torch.cat([self.feat_old(x)['features'], self.feat_new(x)['features']], 1)
        return features

    def forward(self, x):
        out_old = self.feat_old(x)
        out_new = self.feat_new(x)
        F_old = out_old['features']
        F_new = out_new['features']
        F = torch.cat([F_old, F_new], 1)
        out = self.fc(F)
        out.update({'features':F, 'F_new':F_new})
        o_old = self.old_fc(F_old)["logits"]
        out.update(old_logits=o_old)
        if(self.use_auxfc):
            o_aux = self.aux_fc(F_new)["logits"] if self.aux_fc is not None else None
            out.update(aux_logits=o_aux)
        return out


    def update_model(self, compressedNet, nb_classes):
        feat_extractor = compressedNet.convnet
        fc = compressedNet.fc
        #copy net feature extractor in F_old and F_new
        self.update_feats(feat_extractor)
        #add new outputs to fc for new classes
        self.update_fc(fc, nb_classes)
    def update_fc(self, prev_fc, nb_classes):
        self.old_fc = copy.deepcopy(prev_fc)
        fc = self.generate_fc(self.feature_dim, nb_classes)
        prev_out, prev_in = prev_fc.out_features, prev_fc.in_features
        weight = copy.deepcopy(prev_fc.weight.data)
        bias = copy.deepcopy(prev_fc.bias.data)
        fc.weight.data[:prev_out,:prev_in] = weight
        fc.bias.data[:prev_out] = bias
        self.fc = fc
        if(self.use_auxfc):
            new_task_size = nb_classes - prev_out
            self.aux_fc=self.generate_fc(self.out_dim,new_task_size+1)

    def update_feats(self, feat_extractor):
        self.feat_old = copy.deepcopy(feat_extractor)
        self.feat_new = copy.deepcopy(feat_extractor)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
    def partial_freeze(self):
        #freeze only old feature extractor
        for param in self.feat_old.parameters():
            param.requires_grad = False
        for param in self.old_fc.parameters():
            param.requires_grad = False
        self.feat_old.eval()
        self.feat_new.train()
        return self

class DualFeatureNetExp(nn.Module):
    def __init__(self, convnet_type, pretrained, use_auxfc=True):
        super(DualFeatureNet,self).__init__()
        self.convnet_type=convnet_type
        self.pretrained=pretrained
        self.feat_old = get_convnet(self.convnet_type)
        self.feat_new = get_convnet(self.convnet_type)
        self.out_dim=self.feat_old.out_dim
        self.fc = None
        self.aux_fc=None
        self.use_auxfc = use_auxfc
        self.expansion_mode = True

    @property
    def feature_dim(self):
        return self.out_dim*2

    def extract_vector(self, x):
        features = torch.cat([self.feat_old(x)['features'], self.feat_new(x)['features']], 1)
        return features

    def forward(self, x):
        if(not self.expansion_mode):#forward comrpession
            out_old = self.feat_old(x)
            out_new = self.feat_new(x)
            F_old = out_old['features']
            F_new = out_new['features']
            F = torch.cat([F_old, F_new], 1)
            out = self.fc(F_old, F_new)
            out.update({'features':F, 'F_new':F_new})
            if(self.use_auxfc):
                o_aux = self.aux_fc(F_new)["logits"] if self.aux_fc is not None else None
                out.update(aux_logits=o_aux)
        else:#forward epansion
            F_new = self.feat_new(x)['features']
            out = {'features':F_new}
            if(self.use_auxfc):
                o_aux = self.aux_fc(F_new)["logits"] if self.aux_fc is not None else None
            out.update(aux_logits=o_aux)
        return out


    def update_model(self, compressedNet, nb_classes):
        feat_extractor = compressedNet.convnet
        fc = compressedNet.fc
        #copy net feature extractor in F_old and F_new
        self.update_feats(feat_extractor)
        #add new outputs to fc for new classes
        self.update_fc(fc, nb_classes)
    def update_fc(self, prev_fc, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        prev_out, prev_in = prev_fc.out_features, prev_fc.in_features
        weight = copy.deepcopy(prev_fc.weight.data)
        bias = copy.deepcopy(prev_fc.bias.data)
        fc.weight.data[:prev_out,:prev_in] = weight
        fc.bias.data[:prev_out] = bias
        self.fc = fc
        if(self.use_auxfc):
            new_task_size = nb_classes - prev_out
            self.aux_fc=self.generate_fc(self.out_dim,nb_classes)

    def update_feats(self, feat_extractor):
        self.feat_old = copy.deepcopy(feat_extractor)
        self.feat_new = copy.deepcopy(feat_extractor)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
    def partial_freeze(self):
        #freeze only old feature extractor
        for param in self.feat_old.parameters():
            param.requires_grad = False
        self.feat_old.eval()
        self.feat_new.train()
        return self

class FOSTERNet(nn.Module):
    def __init__(self, args, pretrained):
        super(FOSTERNet, self).__init__()
        self.convnet_type = args.convnet_type
        self.convnets = nn.ModuleList()
        self.pretrained = pretrained
        self.out_dim = None
        self.fc = None
        self.fe_fc = None
        self.task_sizes = []
        self.oldfc = None
        self.args = args

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim * len(self.convnets)

    def extract_vector(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        return features

    def forward(self, x):
        features = [convnet(x)["features"] for convnet in self.convnets]
        features = torch.cat(features, 1)
        out = self.fc(features)
        fe_logits = self.fe_fc(features[:, -self.out_dim :])["logits"]

        out.update({"fe_logits": fe_logits, "features": features})

        if self.oldfc is not None:
            old_logits = self.oldfc(features[:, : -self.out_dim])["logits"]
            out.update({"old_logits": old_logits})

        out.update({"eval_logits": out["logits"]})
        return out

    def update_fc(self, nb_classes):
        self.convnets.append(get_convnet(self.convnet_type))
        if self.out_dim is None:
            self.out_dim = self.convnets[-1].out_dim
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output, : self.feature_dim - self.out_dim] = weight
            fc.bias.data[:nb_output] = bias
            self.convnets[-1].load_state_dict(self.convnets[-2].state_dict())

        self.oldfc = self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.fe_fc = self.generate_fc(self.out_dim, nb_classes)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def freeze_conv(self):
        for param in self.convnets.parameters():
            param.requires_grad = False
        self.convnets.eval()

    def weight_align(self, old, increment, value):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew * (value ** (old / increment))
        logging.info("align weights, gamma = {} ".format(gamma))
        self.fc.weight.data[-increment:, :] *= gamma

class MAFDRC_CIFAR(BaseNet):

    def __init__(self, convnet_type, pretrained, gradcam=False,scale=1):
        super().__init__(convnet_type, pretrained)
        self.convnet_type = convnet_type
        self.gradcam = gradcam
        nc = [16,32,64]
        nc = [c*scale for c in nc]
        # 1x1 conv can be replaced with more light-weight bn layer
        self.BHO = nn.BatchNorm2d(nc[2])
        self.BHN = nn.BatchNorm2d(nc[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if hasattr(self, 'gradcam') and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes,cur_task):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

            if cur_task==1:
                self.BHO.weight.data = copy.deepcopy(self.BHN.weight.data)
                self.BHO.bias.data = copy.deepcopy(self.BHN.bias.data)
            else:
                self.BHO.weight.data = (copy.deepcopy(self.BHO.weight.data)+copy.deepcopy(self.BHN.weight.data)) /2
                self.BHO.bias.data = (copy.deepcopy(self.BHO.bias.data)+copy.deepcopy(self.BHN.bias.data)) /2

        del self.fc
        self.fc = fc

    def update_BH(self,BHO,BHN):
        self.BHO.weight.data = copy.deepcopy(BHO.weight.data)
        self.BHO.bias.data = copy.deepcopy(BHO.bias.data)
        self.BHN.weight.data =copy.deepcopy(BHN.weight.data)
        self.BHN.bias.data = copy.deepcopy(BHN.bias.data)

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        new_fs, old_fs = self.BHN(x["half"]), self.BHO(x["half"])
        fs = torch.cat((old_fs,new_fs),dim=0)
        out = self.fc(self.avgpool(fs).view(fs.size(0),-1))
        c = out['logits'].size(0) // 2
        old_logit, new_logit = out["logits"][:c,:], out["logits"][c:,:]
        out.update({"logits":old_logit+new_logit,"old_logits":old_logit,"new_logits":new_logit,"fmaps":x["fmaps"]})
        if hasattr(self, 'gradcam') and self.gradcam:
            out['gradcam_gradients'] = self._gradcam_gradients
            out['gradcam_activations'] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(
            backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(
            forward_hook)

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

class Dualfeat_DRCnet(BaseNet):
    def __init__(self, MAFDRC_net):
        super().__init__(MAFDRC_net.convnet_type, False)
        nb_classes = MAFDRC_net.fc.out_features
        self.feat_old = copy.deepcopy(MAFDRC_net.convnet)
        self.feat_new = copy.deepcopy(MAFDRC_net.convnet)
        self.BHO = nn.BatchNorm2d(self.feature_dim)
        self.BHN = nn.BatchNorm2d(self.feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = SimpleLinear(self.feature_dim, nb_classes)

        #copy branches and fc layer and freeze old feature extractor
        self.copy_BHNO(MAFDRC_net.BHN,MAFDRC_net.BHO)
        self.copy_fc(MAFDRC_net.fc)
        self.freeze_old()
    @property
    def feature_dim(self):
        return self.feat_old.out_dim

    def forward(self, x):
        F_old = self.feat_old(x)["half"]
        F_new = self.feat_new(x)["half"]
        #F = torch.cat([F_old, F_new], 1)
        new_fs, old_fs = self.BHN(F_new), self.BHO(F_old)
        fs = torch.cat((old_fs,new_fs),dim=0)
        feats = self.avgpool(fs).view(fs.size(0),-1)
        out = self.fc(feats)
        c = out['logits'].size(0) // 2
        old_logit, new_logit = out["logits"][:c,:], out["logits"][c:,:]
        out.update({"logits":old_logit+new_logit,"old_logits":old_logit,"new_logits":new_logit,"features":feats})
        return out

    def freeze_old(self):
        #freeze old feature extractor
        for param in self.feat_old.parameters():
            param.requires_grad = False
        return self

    def copy_fc(self, fc):
        weight = copy.deepcopy(fc.weight.data)
        bias = copy.deepcopy(fc.bias.data)
        n, m = weight.shape[0], weight.shape[1]
        self.fc.weight.data[:n, :m] = weight
        self.fc.bias.data[:n] = bias

    def copy_BHNO(self, BH_N, BH_O):
        BHN_w, BHN_b = copy.deepcopy(BH_N.weight.data), copy.deepcopy(BH_N.bias.data)
        BHO_w, BHO_b = copy.deepcopy(BH_O.weight.data), copy.deepcopy(BH_O.bias.data)
        n = BHN_w.shape[0]
        #self.BHN.weight.data[:n] = BHN_w
        #self.BHN.bias.data[:n] = BHN_b
        self.BHO.weight.data[:n] = (BHO_w+BHN_w)/2
        self.BHO.bias.data[:n] = (BHO_b+BHN_b)/2

class SimpleCosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained):
        super().__init__(convnet_type, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data=self.fc.sigma.data
            if nextperiod_initialization is not None:

                weight=torch.cat([weight,nextperiod_initialization])
            fc.weight=nn.Parameter(weight)
        del self.fc
        self.fc = fc


    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class CosineIncrementalNet(BaseNet):

    def __init__(self, convnet_type, pretrained, nb_proxy=1):
        super().__init__(convnet_type, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data

        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            # prev_out_features = self.fc.out_features
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)

        return fc
