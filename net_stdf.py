import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import deform_conv2d


# ==========
# Spatio-temporal deformable fusion module
# ==========


class STDF_Features(nn.Module):
    def __init__(self, in_nc=5, out_nc=64, nf=32, nb=3, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STDF_Features, self).__init__()
        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.base_ks = base_ks

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_nc, 8, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )

        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(8, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )

        self.offset = nn.Conv2d(nf, 2 * self.size_dk * 8, base_ks, padding=base_ks//2)
        self.mask = nn.Conv2d(nf, self.size_dk * 8, base_ks, padding=base_ks//2)
        self.weights = nn.parameter.Parameter(torch.rand(out_nc, 1, self.deform_ks, self.deform_ks))

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        inputs = self.initial_conv(inputs)
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )
        out = self.out_conv(out)
        # compute offset and mask
        # offset: conv offset
        # mask: confidence
        offset = self.offset(out)
        mask = torch.sigmoid(self.mask(out))
        # perform deformable convolutional fusion
        fused_feat = F.relu(
            deform_conv2d(input=inputs, offset=offset, weight=self.weights, padding=self.base_ks // 2, mask=mask),
            inplace=True
        )
        return fused_feat


# ==========
# Quality enhancement and super resolution module module
# ==========


class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=32, nb=8, out_nc=1, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
        )

        hid_conv_lst_1 = []
        for _ in range(nb - 2):
            hid_conv_lst_1 += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
            ]
        self.hid_conv_1 = nn.Sequential(*hid_conv_lst_1)

        self.up1 = nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1)

        hid_conv_lst_2 = []
        for _ in range(nb - 2):
            hid_conv_lst_2 += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
            ]
        self.hid_conv_2 = nn.Sequential(*hid_conv_lst_2)

        self.up2 = nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv_1(out)
        out = self.up1(out)
        out = self.hid_conv_2(out)
        out = self.up2(out)
        out = self.out_conv(out)
        return out


# ==========
# MFVQE_SR_Y network
# ==========


class MFVQE_SR_Y(nn.Module):

    def __init__(self, tmp=5):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE_SR_Y, self).__init__()

        self.ffnet = STDF_Features(tmp)
        self.qenet = PlainCNN()
        self.tmp = tmp

    def forward(self, x):
        out = self.ffnet(x)
        out = self.qenet(out)
        out += torch.nn.functional.interpolate(x[:, self.tmp//2:self.tmp//2+1, ...], scale_factor=4, mode='bicubic',
                                               align_corners=True)  # res: add middle frame
        return out
