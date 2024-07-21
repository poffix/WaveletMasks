import torch
import torch.nn as nn
import torchvision.models as models
import pywt
from utils import comb2
import numpy as np
from pytorch_wavelets import DWTForward, DWTInverse

class Mask(nn.Module):
    """
    nn Layer that computes the Hadamard product 
    between the 2D Fourier transform of the input signal
    and a learnable real-valued mask.
    args mask_size: (height, width)
    """

    def __init__(self, mask_size: tuple = (128, 128)):
        super().__init__()
        assert len(mask_size)==2, "mask_size must be a 2-dim tuple, e.g., (128, 128)"
        
        kernel = torch.Tensor(1, 1, *mask_size)
        self.weight = nn.Parameter(kernel)
        nn.init.ones_(self.weight)
    
    def forward(self, x):
        x = torch.fft.fft2(x)
        x = self.weight * x
        x = torch.fft.ifft2(x).real
        return x 

class WaveletMask(nn.Module):
    """
    nn Layer that computes the Hadamard product 
    between the 2D Wavelet transform of the input signal
    and a learnable real-valued mask.
    args mask_size: (height, width)
    """

    def __init__(self, mask_size: tuple = (128, 128)):
        super().__init__()
        assert len(mask_size)==2, "mask_size must be a 2-dim tuple, e.g., (64, 64)"
        
        kernel = torch.Tensor(1, 1, *mask_size)
        self.weight = nn.Parameter(kernel)
        nn.init.ones_(self.weight)
    
    def forward(self, x):
        # x = x.numpy(force=True)
        # weight = self.weight.clone()
        # weight = weight.numpy(force=True)
        
        # LL, (LH, HL, HH) = pywt.dwt2(x, 'haar')
        # coeffs = weight * np.bmat([[LL.reshape((64,64)), LH.reshape((64,64))], 
        #   [HL.reshape((64,64)), HH.reshape((64,64))]])
        # x = pywt.idwt2((coeffs[:64,:64], (coeffs[:64,64:], coeffs[64:,:64], coeffs[64:,64:])), 'haar').reshape((1,1,128,128))
        # # xfm = DWTForward(wave='haar')
        # # ifm = DWTInverse(wave='haar')
        
        # # if torch.cuda.is_available():
        # #     xfm.cuda()
        # #     ifm.cuda()
        
        # # LL, [B] = xfm(x)
        # # coeffs = weight * np.bmat([[LL.reshape((64,64)), LH.reshape((64,64))], 
        # #   [HL.reshape((64,64)), HH.reshape((64,64))]])
        # # x = pywt.idwt2((coeffs[:64,:64], (coeffs[:64,64:], coeffs[64:,:64], coeffs[64:,64:])), 'haar').reshape((1,1,128,128))

        # return torch.from_numpy(x).cuda() if torch.cuda.is_available() else torch.from_numpy(x)
        xfm = DWTForward(wave='haar')
        ifm = DWTInverse(wave='haar')
        if torch.cuda.is_available():
            xfm.cuda()
            ifm.cuda()
        
        A, [B] = xfm(x)
        A = torch.reshape(A, (64,64))
        B = torch.reshape(B,(3,64,64))
        # B = torch.reshape(B,(3,64,64)).cpu()
        # A = self.weight.cpu() * torch.from_numpy(np.bmat([[torch.reshape(A, (64,64)).cpu(), B[0]], [B[1], B[2]]]))
        l1 = tuple(torch.cat((A[i],B[0][i])) for i in range(A.shape[0]))
        l2 = tuple(torch.cat((B[1][i],B[2][i])) for i in range(A.shape[0]))
        A = self.weight * torch.reshape(torch.cat(l1 + l2),(1,1,A.shape[0]*2,A.shape[1]*2))
        # B = torch.reshape(torch.cat((torch.reshape(A[0,0,:64,64:],(1,64,64)), torch.reshape(A[0,0,64:,:64],(1,64,64)), 
        #                              torch.reshape(A[0,0,64:,64:],(1,64,64)))),(1,1,3,64,64)).cuda()
        # A = torch.reshape(A[0,0,:64,:64],(1,1,64,64)).cuda()
        B = torch.reshape(torch.cat((torch.reshape(A[0,0,:64,64:],(1,64,64)), torch.reshape(A[0,0,64:,:64],(1,64,64)), 
                                     torch.reshape(A[0,0,64:,64:],(1,64,64)))),(1,1,3,64,64))
        A = torch.reshape(A[0,0,:64,:64],(1,1,64,64))
        Y = ifm((A,[B]))
        
        return Y

def Core(
    num_classes, 
    architecture='vgg'
):
    
    """
    VGG11_bn(num_classes) or ResNet18(num_classes) core for grayscale images
    """
    assert architecture in ['vgg', 'resnet']
    
    if architecture=='vgg':
        core = models.vgg11_bn(num_classes=num_classes)
        core.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
    else:
        core = models.resnet18(num_classes=num_classes)
        core.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return core

class MaskCore(nn.Module):
    def __init__(
        self,
        mask_size, 
        num_classes, 
        architecture,
        pretrained_model=None,
        pretrained_mask=None
    ):
        super().__init__()
        
        # self.mask = Mask(mask_size=mask_size)
        self.mask = WaveletMask(mask_size=mask_size)
        self.core = Core(num_classes=num_classes, architecture=architecture)
        
        if pretrained_model is not None:
            assert isinstance(pretrained_model, str), 'if not None, pretrained_model should be the location of a pretrained model'
            checkpoint = torch.load(pretrained_model, map_location="cpu")
            self.core.load_state_dict(checkpoint['state_dict'])
        
        if pretrained_mask is not None:
            assert isinstance(pretrained_mask, str), 'if not None, pretrained_mask should be the location of a pretrained mask'
            self.mask.weight.data = torch.load(pretrained_mask, map_location="cpu")
            
    def forward(self, x):
        x = self.mask(x)
        x = self.core(x)
        return x
    
class LC(torch.nn.Module):
    def __init__(self, img_size, num_classes=5):
        super().__init__()
        self.linear = nn.Linear(img_size[0]*img_size[1], num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.linear(x)
    
class VMask(nn.Module):
    
    """
    nn Layer that computes the Hadamard product 
    between the 2D Fourier transform of the input signal
    and a learnable real-valued mask.
    args mask_size: (height, width)
    """

    def __init__(self, mask_size: tuple = (128, 128), num_masks: int = 1):
        super().__init__()
        assert len(mask_size)==2, "mask_size must be a 2-dim tuple, e.g., (128, 128)"
        
        kernel = torch.Tensor(1, num_masks, *mask_size)
        self.weight = nn.Parameter(kernel)
        nn.init.ones_(self.weight)
    
    def forward(self, x):
        signal_fourier = torch.fft.fft2(x)
        mask_fourier = self.weight * signal_fourier
        inv = torch.fft.ifft2(mask_fourier).real
        return inv.view(inv.shape[0]*inv.shape[1], 1, inv.shape[-2], inv.shape[-1])

class Vote(nn.Module):
    def __init__(self, num_votes: int = 1):
        super().__init__()
        self.num_votes = num_votes
    
    def forward(self, x, y=None):
        x = x.view(x.shape[0]//self.num_votes, self.num_votes, -1)
        if y is not None:
            return x[range(x.shape[0]), y]
        else:
            x_bar = x.transpose(-2, -1).reshape(x.shape[0], 1, -1)
            idx = torch.argmax(x_bar, dim=2) % self.num_votes
            idx = idx.squeeze()
            return x[range(x.shape[0]), idx]
        
class MaskCoreVote(nn.Module):
    def __init__(self, mask_size, num_masks, num_classes, pretrained=None):
        super().__init__()
        
        self.mask = Mask(mask_size, num_masks)
        self.core = Core(num_classes)
        self.vote = Vote(num_masks)
        
        if pretrained is not None:
            assert isinstance(pretrained, str), 'if not None, pretrained should be the location of a pretrained model'
            checkpoint = torch.load(pretrained, map_location="cpu")
            self.core.load_state_dict(checkpoint['state_dict'])
        
    def forward(self, x, mask_y=None):
        x = self.mask(x)
        x = self.core(x)
        x = self.vote(x, mask_y)
        return x
