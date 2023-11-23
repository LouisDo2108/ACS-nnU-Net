# from .functional import acs_conv_f

from .base_acsconv import _ACSConv
import math
import torch.nn.functional as F
import torch

class ACSConv(_ACSConv):
    """
    Vallina ACS Convolution
    
    Args:
        acs_kernel_split: optional, equally spit if not specified.

        Other arguments are the same as torch.nn.Conv3d.
    Examples:
        >>> import ACSConv
        >>> x = torch.rand(batch_size, 3, D, H, W)
        >>> conv = ACSConv(3, 10, kernel_size=3, padding=1)
        >>> out = conv(x)

        >>> conv = ACSConv(3, 10, acs_kernel_split=(4, 3, 3))
        >>> out = conv(x)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, acs_kernel_split=None, 
                 bias=True, padding_mode='zeros'):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, 0, groups, bias, padding_mode)
        if acs_kernel_split is None:
            if self.out_channels%3==0:
                self.acs_kernel_split = (self.out_channels//3,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==1:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3,self.out_channels//3)
            if self.out_channels%3==2:
                self.acs_kernel_split = (self.out_channels//3+1,self.out_channels//3+1,self.out_channels//3)
        else:
            self.acs_kernel_split = acs_kernel_split


    def forward(self, x):
        """
        Convolution forward function
        Divide the kernel into three parts on output channels based on acs_kernel_split, 
        and conduct convolution on three directions seperately. Bias is added at last.
        """
        return self.acs_conv_f(x, self.weight, self.bias, self.kernel_size, self.dilation, self.padding, self.stride, 
                            self.groups, self.out_channels, self.acs_kernel_split)


    def extra_repr(self):
        s = super().extra_repr() + ', acs_kernel_split={acs_kernel_split}'
        return s.format(**self.__dict__)
    
    def conv3D_output_shape_f(self, i, input_shape, kernel_size, dilation, padding, stride):
        """
        Calculate the original output size assuming the convolution is nn.Conv3d based on 
        input size, kernel size, dilation, padding and stride.
        """
        return math.floor((input_shape[i]-kernel_size[i]-(dilation[i]-1)*
                                            (kernel_size[i]-1)+2*padding[i])
                                        /stride[i])+1
        

    def acs_conv_f(self, x, weight, bias, kernel_size, dilation, padding, stride, groups, out_channels, acs_kernel_split):
        B, C_in, *input_shape = x.shape
        C_out = weight.shape[0]
        assert groups==1 or groups==C_in==C_out, "only support standard or depthwise conv"

        conv3D_output_shape = (self.conv3D_output_shape_f(0, input_shape, kernel_size, dilation, padding, stride), 
                                self.conv3D_output_shape_f(1, input_shape, kernel_size, dilation, padding, stride), 
                                self.conv3D_output_shape_f(2, input_shape, kernel_size, dilation, padding, stride))

        weight_a = weight[0:acs_kernel_split[0]].unsqueeze(2)
        weight_c = weight[acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])].unsqueeze(3)
        weight_s = weight[(acs_kernel_split[0]+acs_kernel_split[1]):].unsqueeze(4)
        if groups==C_in==C_out:
            # depth-wise
            x_a = x[:, 0:acs_kernel_split[0]]
            x_c = x[:, acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])]
            x_s = x[:, (acs_kernel_split[0]+acs_kernel_split[1]):]
            group_a = acs_kernel_split[0]
            group_c = acs_kernel_split[1]
            group_s = acs_kernel_split[2]
        else:
            # groups=1
            x_a = x_c = x_s = x
            group_a = group_c = group_s = 1

        f_out = []
        if acs_kernel_split[0]>0:
            a = F.conv3d(x_a if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,padding[0],padding[0]),'constant',0)[:,:,
                                                kernel_size[0]//2:kernel_size[0]//2+(conv3D_output_shape[0]-1)*stride[0]+1,
                                                :,:], 
                                                weight=weight_a, bias=None, 
                                                stride=stride,
                                                padding=(0,padding[1],padding[2]),
                                                dilation=dilation,
                                                groups=group_a)                
            f_out.append(a)
        if acs_kernel_split[1]>0:
            c = F.conv3d(x_c if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,padding[1],padding[1]),'constant',0)[:,:,:,
                                                kernel_size[1]//2:kernel_size[1]//2+stride[1]*(conv3D_output_shape[1]-1)+1,
                                                :], 
                                                weight=weight_c, bias=None,                                     
                                                stride=stride,
                                                padding=(padding[0],0,padding[2]),
                                                dilation=dilation,
                                                groups=group_c)
            f_out.append(c)
        if acs_kernel_split[2]>0:
            s = F.conv3d(x_s if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (padding[2],padding[2]),'constant',0)[:,:,:,:,
                                                kernel_size[2]//2:kernel_size[2]//2+stride[2]*(conv3D_output_shape[2]-1)+1
                                                ], 
                                                weight=weight_s, 
                                                bias=None, 
                                                stride=stride,
                                                padding=(padding[0],padding[1],0),
                                                dilation=dilation,
                                                groups=group_s)
            f_out.append(s)
        f = torch.cat(f_out, dim=1)
        
        if bias is not None:
            f += bias.view(1,out_channels,1,1,1)

        return f

