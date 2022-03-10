import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import copy

from ..attack import Attack

class OPTA(Attack):
    r"""
    Mixed Translation Optimization Attacks
    Leverages Momentum FGSM idea while trying different optimizations.

    Distance Measure : L2

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

    def __init__(self, model, data_loader, eps=8/255, alpha=2/255, steps=5, decay=1.0, model_extra=None):
        super().__init__("OPTA", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self._supported_mode = ['default', 'targeted']
        self.data_loader = data_loader
        self.total_filters = (model.conv1.out_channels+model.conv2.out_channels) #TODO: needs to be done automated
        self.num_classes = 10 #assume MNIST TODO: make variable
        self.height = 28 #assume MNIST
        self.width = 28 #assume MNIST
        self.channels = 1 #assume MNIST
        self.model_extra = model_extra

    def _feature_l1_norm(self, f1, f2):
        return torch.sum(torch.abs(f1-f2))
    def _feature_l2_norm(self, f1, f2):
        return torch.sum((f1-f2)**2)

    def set_grad_translation(self, x):
        o_1, z1_1, a1_1, z2_1, a2_1 = self.model.semi_forward(x[:1])
        o_2, z1_2, a1_2, z2_2, a2_2 = self.model.semi_forward(x[1:])

        alpha = 1
        beta = 1

        cost = alpha*self._feature_l2_norm(z1_1, z1_2) + beta*self._feature_l2_norm(z2_1, z2_2) #TODO: Make generalized for all networks
        grad_x = torch.autograd.grad(cost, x,
               retain_graph=True, create_graph=True)[0]

        return grad_x, cost

    def set_grad_mixing(self, x, x_orig):
        o_1, z1_1, a1_1, z2_1, a2_1 = self.model.semi_forward(x[:1])
        o_2, z1_2, a1_2, z2_2, a2_2 = self.model.semi_forward(x[1:])
        o_3, z1_3, a1_3, z2_3, a2_3 = self.model.semi_forward(x_orig[:1])

        alpha = 1
        beta = 0

        cost = alpha*self._feature_l2_norm(z1_1, z1_2) + beta*self._feature_l2_norm(z1_1, z1_3)
        grad_x = torch.autograd.grad(cost, x, 
                retain_graph=True, create_graph=True)[0]

        return grad_x, cost

    def set_final_distribution(self, x, labels, target_labels, distribution):
        output_c = self.model(x)
        probs_c = self.model_extra(output_c)

        ideal_distribution = torch.zeros(self.num_classes)
        ideal_distribution[labels[0]] = distribution[0]
        ideal_distribution[target_labels[0]] = distribution[1]

        # print (probs_c)
        # print (ideal_distribution)

        kl_loss = nn.KLDivLoss(log_target=False)
        cost = kl_loss(probs_c[0], ideal_distribution)

        grad_x = torch.autograd.grad(cost, x, retain_graph=True, create_graph=True)[0]

        return grad_x, cost

    def update_image_with_grad_translation(self, x_star, x, grad):
        x_star = x_star.detach() - self.alpha*grad.sign()
        delta = torch.clamp(x_star - x, min=-self.eps, max=self.eps)
        adv_images = torch.clamp(x + delta, min=0, max=1).detach()

        return adv_images

    def get_random_target_label(self, labels):
        b_size = labels.shape[0]
        targets = torch.randint(0, self.num_classes, (b_size, ))

        return targets

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            #target_labels = self._get_target_label(images, labels)
            target_labels = self.get_random_target_label(labels)
            print (labels)
            print (target_labels)

        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True

            #for now intercept here
            #grad, cost = self.set_grad_translation(adv_images)
            #grad, cost = self.set_grad_mixing(adv_images, images)
            distribution = [0.5, 0.5]
            grad, cost = self.set_final_distribution(adv_images, labels, target_labels, distribution)
            print (cost)

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = self.update_image_with_grad_translation(adv_images, images, grad)
            #adv_images[1:] = images[1:] #reset adv_images target

        return adv_images