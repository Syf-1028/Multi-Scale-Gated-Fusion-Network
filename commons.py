"""
 > Common/standard network archutectures and modules
 > Credit for some functions
    * github.com/eriklindernoren/PyTorch-GAN
    * pluralsight.com/guides/artistic-neural-style-transfer-with-pytorch
 > Maintainer: https://github.com/xahidbuffon
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
from torchvision.models import vgg19


def Weights_Normal(m):
    # initialize weights as Normal(mean, std)
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)


class UNetDown(nn.Module):
    """ Standard UNet down-sampling block 
    """
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """ Standard UNet up-sampling block
    """
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class VGG19_PercepLoss(nn.Module):
    """ Calculates perceptual loss in vgg19 space
    """
    def __init__(self, _pretrained_=True):
        super(VGG19_PercepLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=_pretrained_).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

    def get_features(self, image, layers=None):
        if layers is None: 
            layers = {'30': 'conv5_2'} # may add other layers
        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

    def forward(self, pred, true, layer='conv5_2'):
        true_f = self.get_features(true)
        pred_f = self.get_features(pred)
        return torch.mean((true_f[layer]-pred_f[layer])**2)


# class Gradient_Penalty(nn.Module):
#     """ Calculates the gradient penalty loss for WGAN GP
#     """
#     def __init__(self, cuda=True):
#         super(Gradient_Penalty, self).__init__()
#         self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
#     def forward(self, D, real, fake):
#         # Random weight term for interpolation between real and fake samples
#         eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
#         # Get random interpolation between real and fake samples
#         interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)
#         d_interpolates = D(interpolates)
#         fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
#         # Get gradient w.r.t. interpolates
#         gradients = autograd.grad(outputs=d_interpolates,
#                                   inputs=interpolates,
#                                   grad_outputs=fake,
#                                   create_graph=True,
#                                   retain_graph=True,
#                                   only_inputs=True,)[0]
#         gradients = gradients.view(gradients.size(0), -1)
#         gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
#         return gradient_penalty

class Gradient_Penalty(nn.Module):
    """ Calculates the gradient penalty loss for WGAN GP
    """

    def __init__(self, cuda=True):
        super(Gradient_Penalty, self).__init__()
        self.Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    def forward(self, D, real, fake, distorted=None):  # 添加 distorted 参数
        # Random weight term for interpolation between real and fake samples
        eps = self.Tensor(np.random.random((real.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (eps * real + ((1 - eps) * fake)).requires_grad_(True)

        # ============ 关键修改 ============
        if distorted is not None:
            # 如果有条件图像，传入两个参数
            d_interpolates = D(interpolates, distorted)
        else:
            # 否则只传一个（保持向后兼容）
            d_interpolates = D(interpolates)
        # ================================

        fake = autograd.Variable(self.Tensor(d_interpolates.shape).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(outputs=d_interpolates,
                                  inputs=interpolates,
                                  grad_outputs=fake,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True, )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

class VGG19ForPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGG19ForPerceptualLoss, self).__init__()
        # 修复：使用新的weights参数替代弃用的pretrained参数
        vgg = vgg19(weights='DEFAULT').features  # 或者使用 weights='IMAGENET1K_V1'

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        # 分层提取特征
        for x in range(4):  # conv1
            self.slice1.add_module(str(x), vgg[x])
        for x in range(4, 9):  # conv2
            self.slice2.add_module(str(x), vgg[x])
        for x in range(9, 18):  # conv3, conv4
            self.slice3.add_module(str(x), vgg[x])
        for x in range(18, 27):  # conv5
            self.slice4.add_module(str(x), vgg[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        """计算x和y在VGG特征空间上的感知损失"""
        # 归一化到VGG的输入范围
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x + 1) / 2  # 从[-1,1]到[0,1]
        x = (x - mean) / std
        y = (y + 1) / 2
        y = (y - mean) / std

        # 提取多层特征并计算损失
        h_x = self.slice1(x)
        h_y = self.slice1(y)
        h1_loss = F.l1_loss(h_x, h_y)

        h_x = self.slice2(h_x)
        h_y = self.slice2(h_y)
        h2_loss = F.l1_loss(h_x, h_y)

        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        h3_loss = F.l1_loss(h_x, h_y)

        h_x = self.slice4(h_x)
        h_y = self.slice4(h_y)
        h4_loss = F.l1_loss(h_x, h_y)

        return h1_loss + h2_loss + h3_loss + h4_loss

class EnhancedLoss(nn.Module):
    """
    修改3: 创建组合损失函数
    整合相对论GAN损失、L1损失、感知损失和梯度损失
    """

    def __init__(self, lambda_l1=100.0, lambda_perceptual=1.0, lambda_gradient=1.0, device='cuda'):
        super(EnhancedLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_gradient = lambda_gradient

        # L1 损失
        self.l1_loss = nn.L1Loss()

        # 感知损失
        self.vgg_loss = VGG19ForPerceptualLoss().to(device)

        # Sobel梯度算子
        self.sobel_x, self.sobel_y = self._get_sobel_kernels(device)

    def _get_sobel_kernels(self, device):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=device).view(1, 1, 3, 3)
        return sobel_x, sobel_y

    def gradient_loss(self, pred, target):
        """计算梯度损失 - 修复版本"""
        # 扩展Sobel核以匹配输入通道数
        sobel_x = self.sobel_x.repeat(pred.shape[1], 1, 1, 1)  # [3, 1, 3, 3]
        sobel_y = self.sobel_y.repeat(pred.shape[1], 1, 1, 1)  # [3, 1, 3, 3]

        # 预测图像的梯度
        grad_pred_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.shape[1])
        grad_pred_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.shape[1])

        # 真实图像的梯度
        grad_target_x = F.conv2d(target, sobel_x, padding=1, groups=target.shape[1])
        grad_target_y = F.conv2d(target, sobel_y, padding=1, groups=target.shape[1])

        # 计算L1损失
        loss = F.l1_loss(grad_pred_x, grad_target_x) + F.l1_loss(grad_pred_y, grad_target_y)
        return loss

    def relativistic_gan_loss(self, d_real, d_fake, for_generator=True):
        """计算相对论GAN损失"""
        if for_generator:
            # 生成器损失：让假图像看起来比真图像更真实
            real_loss = F.binary_cross_entropy_with_logits(
                d_real - d_fake.mean(0, keepdim=True), torch.zeros_like(d_real))
            fake_loss = F.binary_cross_entropy_with_logits(
                d_fake - d_real.mean(0, keepdim=True), torch.ones_like(d_fake))
        else:
            # 判别器损失：让真图像看起来比假图像更真实
            real_loss = F.binary_cross_entropy_with_logits(
                d_real - d_fake.mean(0, keepdim=True), torch.ones_like(d_real))
            fake_loss = F.binary_cross_entropy_with_logits(
                d_fake - d_real.mean(0, keepdim=True), torch.zeros_like(d_fake))

        return (real_loss + fake_loss) / 2

    def forward(self, discriminator, gen_output, target, real_input, fake_input, for_generator=True):
        """
        计算组合损失
        修改：适配 DiscriminatorFunieGAN 的接口 (img_A, img_B)
        """
        total_loss = 0
        losses = {}

        # ============ 修改点：正确调用判别器 ============
        # 原代码：d_real = discriminator(real_input, target)
        # 改为：传入两个参数 (img, distorted)
        d_real = discriminator(real_input, target)  # real_input是真实图像，target是条件（失真图像）
        d_fake = discriminator(fake_input, target)  # fake_input是生成图像，target是条件（失真图像）
        # ==============================================

        gan_loss = self.relativistic_gan_loss(d_real, d_fake, for_generator)
        total_loss += gan_loss
        losses['GAN'] = gan_loss

        if for_generator:
            # 2. L1损失
            l1_l = self.l1_loss(gen_output, target) * self.lambda_l1
            total_loss += l1_l
            losses['L1'] = l1_l

            # 3. 感知损失
            if self.lambda_perceptual > 0:
                perc_l = self.vgg_loss(gen_output, target) * self.lambda_perceptual
                total_loss += perc_l
                losses['Perceptual'] = perc_l

            # 4. 梯度损失
            if self.lambda_gradient > 0:
                grad_l = self.gradient_loss(gen_output, target) * self.lambda_gradient
                total_loss += grad_l
                losses['Gradient'] = grad_l

            losses['Total'] = total_loss

        return total_loss, losses