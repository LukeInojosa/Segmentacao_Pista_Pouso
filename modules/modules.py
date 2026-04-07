from torch import nn
import torch
import numpy as np


class DepthwiseConvolution(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size = 3):
    super(DepthwiseConvolution,self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size= kernel_size, padding = (kernel_size-1)//2, groups = in_channels),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(inplace = True),
        nn.Conv2d(in_channels, out_channels, kernel_size= 1)
    )
  def forward(self,x):
    return self.conv(x)
  

class DoubleConvolution(nn.Module):
  def __init__(self, in_channels,out_channels, kernel_size = 3, mid_channels = None, efficient = False):
    super(DoubleConvolution,self).__init__()
    if not mid_channels:
      mid_channels = out_channels
    if not efficient:
      self.conv = nn.Sequential(
          nn.Conv2d(in_channels, mid_channels , kernel_size= kernel_size, padding = (kernel_size-1)//2),
          nn.BatchNorm2d(mid_channels),
          nn.LeakyReLU(0.01, inplace=True),
          nn.Conv2d(mid_channels, out_channels, kernel_size= kernel_size, padding = (kernel_size-1)//2),
          nn.BatchNorm2d(out_channels),
          nn.LeakyReLU(0.01, inplace=True),
      )
    else:
      self.conv = nn.Sequential(
        DepthwiseConvolution(in_channels, mid_channels, kernel_size),
        nn.BatchNorm2d(mid_channels),
        nn.LeakyReLU(0.01, inplace=True),
        DepthwiseConvolution(mid_channels, out_channels, kernel_size),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.01, inplace=True),
      )
  def forward(self,x):
    return self.conv(x)
  


# Duplica tamanho da imagem e divide pela metade a quantidade de canais
class Up(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               mid_channels = None,
               image_scale_factor = 2,
               light = False,
               batch_normalization = True,
               activation_function = nn.LeakyReLU(0.01, inplace=True)):
    super(Up, self).__init__()

    self.light = light

    if not mid_channels:
      mid_channels = out_channels

    modules = []
    if light:
      modules = [
          nn.Upsample(scale_factor= image_scale_factor, mode='bilinear', align_corners=False),
          nn.Conv2d(in_channels, mid_channels, kernel_size=1)
      ]
    else:
      modules = [
          nn.ConvTranspose2d(in_channels, mid_channels, kernel_size= image_scale_factor,stride = image_scale_factor)
      ]

    if batch_normalization:
          modules.append(nn.BatchNorm2d(mid_channels))

    modules.append(activation_function)

    self.up_sample = nn.Sequential(*modules)

    self.conv = DoubleConvolution(in_channels=mid_channels,
                            out_channels=out_channels,
                            efficient=True)

  def forward(self,x):
    x = self.up_sample(x)
    return self.conv(x)
  
class ViT_Encoder(nn.Module):
    def __init__(self,vit):
        super(ViT_Encoder, self).__init__()
        self.vit = vit
    def forward(self,x):
        # Reshape and permute the input tensor
        x = self.vit._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # # Classifier "token" as used by standard language architectures
        class_token = x[:, 0]
        x = x[:, 1:]
        batch_size, seq_length_square, hidden_dim  = x.shape
        H = W = int(seq_length_square ** 0.5)

        x = x.permute(0,2,1)
        x = x.reshape(batch_size,hidden_dim,H, W)

        # x = self.vit.heads(x)

        return x
    

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder,self).__init__()
        self.in_channels = in_channels
        self.up1 = Up(in_channels = self.in_channels,
                      mid_channels= self._update_in_ret_out(self.in_channels//4),
                      out_channels= self._update_in_ret_out(self.in_channels//2),
                      image_scale_factor=2,
                      light= True,)
        self.up2 = Up(in_channels = self.in_channels,
                      mid_channels= self._update_in_ret_out(self.in_channels//4),
                      out_channels= self._update_in_ret_out(self.in_channels//2),
                      image_scale_factor=4,
                      light= True,)
        self.up3 = Up(in_channels = self.in_channels,
                      mid_channels= self._update_in_ret_out(self.in_channels//2),
                      out_channels= self._update_in_ret_out(self.in_channels//2),
                      image_scale_factor=2,
                      light= True)
        self.conv = nn.Sequential(
                      DoubleConvolution(in_channels = (self.in_channels + 3),
                                        out_channels= self._update_in_ret_out(3)),
                      DepthwiseConvolution(in_channels= self.in_channels,
                                          out_channels= self._update_in_ret_out(1),
                                          kernel_size=3)
        )

    def _update_in_ret_out(self,out):
        self.in_channels = out
        return out

    def forward(self,x_encoded,x):
        x_encoded = self.up1(x_encoded)
        x_encoded = self.up2(x_encoded)
        x_encoded = self.up3(x_encoded)
        x = torch.cat([x_encoded,x], dim=1)
        x = self.conv(x)
        return x

  

