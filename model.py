import torch
import torch.nn as nn


####################
# Basic blocks
####################


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1):
    # helper selecting activation
    # neg_slope: for leakyrelu and init of prelu
    # n_prelu: for p_relu num_parameters
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == 'sigm':
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


def norm(norm_type, nc):
    # helper selecting normalization layer
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    # helper selecting padding layer
    # if padding is 'zero', do by conv layers
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True, \
               pad_type='zero', norm_type=None, act_type='relu', mode='CNA'):
    '''
    Conv layer with padding, normalization, activation
    mode: CNA --> Conv -> Norm -> Act
        NAC --> Norm -> Act --> Conv (Identity Mappings in Deep Residual Networks, ECCV16)
    '''
    assert mode in ['CNA', 'NAC', 'CNAC'], 'Wong conv mode [{:s}]'.format(mode)
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding, \
            dilation=dilation, bias=bias, groups=groups)
    a = act(act_type) if act_type else None
    if 'CNA' in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == 'NAC':
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
            # Important!
            # input----ReLU(inplace)----Conv--+----output
            #        |________________________|
            # inplace ReLU will modify the input, therefore wrong output
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)

def sequential(*args):
    # Flatten Sequential. It unwraps nn.Sequential.
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class VGG_Block(nn.Module):
    def __init__(self, in_nc, out_nc, kernel_size=3, norm_type='batch', act_type='leakyrelu'):
        super(VGG_Block, self).__init__()

        self.conv0 = conv_block(in_nc, out_nc, kernel_size=kernel_size, norm_type=norm_type, act_type=act_type)
        self.conv1 = conv_block(out_nc, out_nc, kernel_size=kernel_size, stride=2, norm_type=None, act_type=act_type)

    def forward(self, x):
        x1 = self.conv0(x)
        out = self.conv1(x1)

        return out


class VGGGAPQualifier(nn.Module):
    def __init__(self, in_nc=3, base_nf=32, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(VGGGAPQualifier, self).__init__()
        # 1024,768,3

        B11 = VGG_Block(in_nc, base_nf, norm_type=norm_type, act_type=act_type)
        # 512,384,32
        B12 = VGG_Block(base_nf, base_nf, norm_type=norm_type, act_type=act_type)
        # 256,192,32
        B13 = VGG_Block(base_nf, base_nf * 2, norm_type=norm_type, act_type=act_type)
        # 128,96,64
        B14 = VGG_Block(base_nf * 2, base_nf * 2, norm_type=norm_type, act_type=act_type)
        # 64,48,64

        # 1024,768,3
        B21 = VGG_Block(in_nc, base_nf, norm_type=norm_type, act_type=act_type)
        # 512,384,32
        B22 = VGG_Block(base_nf, base_nf, norm_type=norm_type, act_type=act_type)
        # 256,192,32
        B23 = VGG_Block(base_nf, base_nf * 2, norm_type=norm_type, act_type=act_type)
        # 128,96,64
        B24 = VGG_Block(base_nf * 2, base_nf * 2, norm_type=norm_type, act_type=act_type)
        # 64,48,64

        B3 = VGG_Block(base_nf * 2, base_nf * 4, norm_type=norm_type, act_type=act_type)
        # 32,24,128
        B4 = VGG_Block(base_nf * 4, base_nf * 8, norm_type=norm_type, act_type=act_type)
        # 16,12,256
        B5 = VGG_Block(base_nf * 8, base_nf * 16, norm_type=norm_type, act_type=act_type)

        self.feature1 = sequential(B11, B12, B13, B14)
        self.feature2 = sequential(B21, B22, B23, B24)

        self.combine = sequential(B3, B4, B5)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # classifie
        self.classifier = nn.Sequential(
            nn.Linear(base_nf * 16, 512), nn.LeakyReLU(0.2, True), nn.Dropout(0.25), nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True), nn.Dropout(0.5), nn.Linear(256, 1), nn.LeakyReLU(0.2, True))

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(x)
        x = self.gap(self.combine(f1 - f2))

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
