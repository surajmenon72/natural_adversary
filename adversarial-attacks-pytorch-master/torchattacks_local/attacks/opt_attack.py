import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import copy

from ..attack import Attack

class OPTA(Attack):
    r"""
    Filter Importance Basic Attack (In Development)
    Leverages Momentum FGSM idea while taking into account a Filter Importance Scaling

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        decay (float): momentum factor. (Default: 1.0)
        steps (int): number of iterations. (Default: 5)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.FIBA(model, eps=8/255, steps=5, decay=1.0)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, data_loader, eps=8/255, alpha=2/255, steps=5, decay=1.0):
        super().__init__("OPTA", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.mode = mode
        self._supported_mode = ['default', 'targeted']
        self.data_loader = data_loader
        self.total_filters = (model.conv1.out_channels+model.conv2.out_channels) #TODO: needs to be done automated
        self.num_classes = 10 #assume mnist TODO: make variable
        self.height = 28 #assume MNIST
        self.width = 28 #assume MNIST
        self.channels = 1 #assume MNIST

    def set_grad(self, x, y, y_targ):


    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        print (target_labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            #for now intercept here
            grad = self.set_grad(adv_images, labels, target_labels)


            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            if (cost == 0):
                break

        return adv_images