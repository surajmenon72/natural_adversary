import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import copy

from ..attack import Attack

class FIBA(Attack):
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

    def __init__(self, model, data_loader, eps=8/255, alpha=2/255, steps=5, decay=1.0, mode='Identity'):
        super().__init__("FIBA", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self.mode = mode
        self._supported_mode = ['default', 'targeted']
        self.data_loader = data_loader
        self.total_filters = (model.conv1.out_channels+model.conv2.out_channels) #TODO: needs to be done automated
        self.fi_dict = torch.zeros(self.total_filters)
        self.fi_dict_rankings = torch.zeros(self.total_filters)
        self.num_classes = 10 #assume mnist TODO: make variable
        self.height = 28 #assume MNIST
        self.width = 28 #assume MNIST
        self.channels = 1 #assume MNIST
        self.fi_dict_class = torch.zeros((self.num_classes, self.total_filters))
        self.fi_dict_class_rankings = torch.zeros((self.num_classes, self.total_filters))
        self.pixel_importance = torch.zeros((self.channels, self.height, self.width))
        self.pixel_importance_rank = torch.zeros((self.channels, self.height, self.width))

    def fi_2_norm(self, val):
        return (val**2)

    def fi_1_norm(self, val):
        return (torch.abs(val))

    def set_fi(self, sample_thresh):
        print ('Setting Filter Importance')
        print (sample_thresh)
        print ('Iterations')
        loss = nn.CrossEntropyLoss()
        iteration = 0
        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True

            if self._targeted:
                target_labels = self._get_target_label(data, target)

            target_labels = target_labels.to(self.device)

            outputs, a, a_a, b, b_a = self.model.semi_forward(data)

            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, target)

            grad_a = torch.autograd.grad(cost, a,
                           retain_graph=True, create_graph=True)[0]


            grad_b = torch.autograd.grad(cost, b,
                           retain_graph=True, create_graph=True)[0]

            #grad_a = torch.squeeze(grad_a)
            #grad_b = torch.squeeze(grad_b)

            f_index = 0
            b, f, h, w = grad_a.shape
            for i in range(f):
                for j in range(h):
                    for k in range(w):
                        self.fi_dict[f_index+i] += self.fi_2_norm(grad_a[0, i, j, k])

            f_index += f
            b, f, h, w = grad_b.shape
            for i in range(f):
                for j in range(h):
                    for k in range(w):
                        self.fi_dict[f_index+i] += self.fi_2_norm(grad_b[0, i, j, k])


            iteration += 1
            if (iteration >= sample_thresh):
                break

        #once all accumulation is done, now we can normalize
        total = torch.sum(self.fi_dict, axis=0)
        for i in range(self.total_filters):
            self.fi_dict[i] /= total

        args = torch.argsort(self.fi_dict, axis=0)

        for i in range(self.total_filters):
            self.fi_dict_rankings[args[i]] = i

        print ('Set Filter Importance Dict and Rankings')
        print (self.fi_dict)
        print (self.fi_dict_rankings)

    def set_class_fi(self, sample_thresh):
        print ('Setting Filter Importance')
        print (sample_thresh)
        print ('Iterations')
        sample_nums = torch.zeros(self.num_classes)
        loss = nn.CrossEntropyLoss()
        iteration = 0
        for data, target in self.data_loader:
            data, target = data.to(self.device), target.to(self.device)
            data.requires_grad = True

            outputs, a, a_a, b, b_a = self.model.semi_forward(data[:1, :])

            cost = loss(outputs, target[:1])

            grad_a = torch.autograd.grad(cost, a,
                           retain_graph=True, create_graph=True)[0]


            grad_b = torch.autograd.grad(cost, b,
                           retain_graph=True, create_graph=True)[0]

            #grad_a = torch.squeeze(grad_a)
            #grad_b = torch.squeeze(grad_b)

            label = int(target[:1].detach().numpy())
            sample_nums[label] += 1

            f_index = 0
            b, f, h, w = grad_a.shape
            for i in range(f):
                for j in range(h):
                    for k in range(w):
                        self.fi_dict_class[label, f_index+i] += self.fi_2_norm(grad_a[0, i, j, k])

            f_index += f
            b, f, h, w = grad_b.shape
            for i in range(f):
                for j in range(h):
                    for k in range(w):
                        self.fi_dict_class[label, f_index+i] += self.fi_2_norm(grad_b[0, i, j, k])


            iteration += 1
            if (iteration >= sample_thresh):
                print ('Class Samples Processed')
                print (sample_nums)
                break

        #once all accumulation is done, now we can normalize
        for k in range(self.num_classes):
            total = torch.sum(self.fi_dict_class[k, :], axis=0)
            for i in range(self.total_filters):
                if (total > 0):
                    self.fi_dict_class[k, i] /= total

        for k in range(self.num_classes):
            args = torch.argsort(self.fi_dict_class[k, :], axis=0)
            for i in range(self.total_filters):
                self.fi_dict_class_rankings[k, args[i]] = i

        print ('Set Filter Importance Dict and Rankings')
        print (self.fi_dict_class)
        print (self.fi_dict_class_rankings)

    def scale_identity(self, val, imp):
        return val

    def scale_linear(self, val, imp, alpha):
        return alpha*imp*val

    def scale_inverse(self, val, imp, alpha):
        return alpha*(1-imp)*val

    def scale_step(self, val, imp, rank, alpha):
        scaled_val = 0
        filter_range_low = int(self.total_filters*(1/3))
        filter_range_high = int(self.total_filters*(2/3))

        filter_range_low = 8
        filter_range_high = 22

        if ((rank > filter_range_low) and (rank < filter_range_high)):
            scaled_val = alpha*val
        else:
            scaled_val = 1e-3*val

        return scaled_val

    def scale_step_high(self, val, imp, rank, alpha):
        scaled_val = 0

        filter_range_low = 27
        filter_range_high = 30

        if ((rank > filter_range_low) and (rank < filter_range_high)):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_step_low(self, val, imp, rank, alpha):
        scaled_val = 0

        filter_range_low = 0
        filter_range_high = 3

        if ((rank > filter_range_low) and (rank < filter_range_high)):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_step_layer_low(self, val, level, alpha):
        scaled_val = 0

        if (level == 1):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_step_layer_high(self, val, level, alpha):
        scaled_val = 0

        if (level == 2):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_gaussian(self, val, imp, alpha):
        #TODO: still to implement
        return val

    def scale_func(self, grad, f_index, layer, label):
        sgn = torch.sign(grad)
        val = torch.abs(grad)
        imp = self.fi_dict_class[label[0], f_index]
        rank = self.fi_dict_class_rankings[label[0], f_index]

        #select scaling func, #TODO: make set in main func
        if (self.mode == 'Identity'):
            scaled_val = self.scale_identity(val, imp)
        elif (self.mode == 'Linear'):
            scaled_val = self.scale_linear(val, imp, 1)
        elif (self.mode == 'Inverse'):
            scaled_val = self.scale_inverse(val, imp, 1)
        elif (self.mode == 'Step-Middle'):
            scaled_val = self.scale_step(val, imp, rank, 1)
        elif (self.mode == 'Step-High'):
            scaled_val = self.scale_step_high(val, imp, rank, 1)
        elif (self.mode == 'Step-Low'):
            scaled_val = self.scale_step_low(val, imp, rank, 1)
        elif (self.mode == 'Step-Layer-Low'):
            scaled_val = self.scale_step_layer_low(val, layer, 1)
        elif (self.mode == 'Step-Layer-High'):
            scaled_val = self.scale_step_layer_high(val, layer, 1)
        else:
            print ('NOT SUPPORTED MODE')
            scaled_val = self.scale_identity(val, imp)

        return (sgn*scaled_val)

    def grad_play(self, x, y, y_targ):
        x, y = x.to(self.device), y.to(self.device)
        x.requires_grad = True
        loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(log_target=True)

        outputs, a, a_a, b, b_a = self.model.semi_forward(x)
        #cost = -loss(outputs, y_targ) #moves towards y_targ (due to ascending gradient)
        #cost = -loss(outputs, y) #moves towards y

        #y_targ = (y_targ+1)%10

        #alpha = 0
        #beta = 1
        #cost = alpha*(-loss(outputs, y)) + beta*(-loss(outputs, y_targ))
        #cost_kl = (kl_loss(outputs[0], outputs[1])) s
        cost = -loss(outputs[:1, :], y_targ[:1])
        #cost_b = loss(outputs[:1, :], y[:1])

        grad_x = torch.autograd.grad(cost, x,
                       retain_graph=True, create_graph=True)[0]

        # grad_x_b = torch.autograd.grad(cost_b, x,
        #                retain_graph=True, create_graph=True)[0]

        grad_a = torch.autograd.grad(cost, a,
                       retain_graph=True, create_graph=True)[0]

        grad_b = torch.autograd.grad(cost, b,
                       retain_graph=True, create_graph=True)[0]

        img_real = x[0].permute(1, 2, 0)
        img_grad = grad_x[0].permute(1, 2, 0)
        #img_grad_b = grad_x_b[0].permute(1, 2, 0)
        img_grad_abs = (torch.abs(img_grad) - img_grad) #seems to give a decent pixel importance
        #img_grad_b_abs = torch.abs(img_grad_b) - img_grad_b

        # f, axarr = plt.subplots(2, 2)
        # axarr[0, 0].imshow(img_real.detach().numpy(), cmap='gray')
        # axarr[0, 1].imshow(img_grad.detach().numpy(), cmap='gray')
        # axarr[1, 0].imshow(img_grad_abs.detach().numpy(), cmap='gray')
        # axarr[1, 1].imshow(img_grad_abs.detach().numpy(), cmap='gray')
        # plt.show()
        # exit()

        #grad_a = torch.squeeze(grad_a)
        #grad_b = torch.squeeze(grad_b)

        #batch, c, h, w = grad_x.shape
        #grad_x = grad_x.view(c, h, w)

        ab, af, ah, aw = grad_a.shape
        bb, bf, bh, bw = grad_b.shape
        xb, xf, xh, xw = grad_x.shape


        #TODO: Set batch size here
        #keep separate for now, but can be combined eventually for memory savings
        final_grads_1 = torch.zeros((1, xf, xh, xw)) #layer 1
        final_grads_2 = torch.zeros((1, xf, xh, xw)) #layer 2

        filter_index = 0
        layer = 1
        for i in range(af):
            for j in range(ah):
                for k in range(aw):
                    scaled_grad = self.scale_func(grad_a[0, i, j, k], filter_index+i, layer, y)
                    s_grad = torch.autograd.grad(a[0, i, j, k], x,
                                retain_graph=True, create_graph=True, allow_unused=True)[0]

                    final_grads_1[0, :, :, :] += scaled_grad * s_grad[0, :, :, :]

        filter_index += af
        layer += 1
        for i in range(bf):
            for j in range(bh):
                for k in range(bw):
                    scaled_grad = self.scale_func(grad_b[0, i, j, k], filter_index+i, layer, y)
                    s_grad = torch.autograd.grad(b[0, i, j, k], x, 
                                retain_graph=True, create_graph=True, allow_unused=True)[0]

                    final_grads_2[0, :, :, :] += scaled_grad * s_grad[0, :, :, :]

        #scale grads by the pixel importance
        #final_grads_1[0, :, :, :] = final_grads_1[0, :, :, :] * self.pixel_importance
        #final_grads_2[0, :, :, :] = final_grads_2[0, :, :, :] * self.pixel_importance

        #for now take midpoint for return grad
        final_grads = (final_grads_1 + final_grads_2)/2
        return final_grads, cost

    def grad_play_pi(self, x, y, y_targ):
        x, y = x.to(self.device), y.to(self.device)
        x.requires_grad = True
        loss = nn.CrossEntropyLoss()
        kl_loss = nn.KLDivLoss(log_target=True)

        outputs, a, a_a, b, b_a = self.model.semi_forward(x)
        #cost = -loss(outputs, y_targ) #moves towards y_targ (due to ascending gradient)
        #cost = -loss(outputs, y) #moves towards y

        #y_targ = (y_targ+1)%10

        #alpha = 0
        #beta = 1
        #cost = alpha*(-loss(outputs, y)) + beta*(-loss(outputs, y_targ))
        #cost_kl = (kl_loss(outputs[0], outputs[1])) s
        cost = -loss(outputs[:1, :], y_targ[:1])
        #cost_b = loss(outputs[:1, :], y[:1])

        grad_x = torch.autograd.grad(cost, x,
                       retain_graph=True, create_graph=True)[0]

        # grad_x_b = torch.autograd.grad(cost_b, x,
        #                retain_graph=True, create_graph=True)[0]

        grad_a = torch.autograd.grad(cost, a,
                       retain_graph=True, create_graph=True)[0]

        grad_b = torch.autograd.grad(cost, b,
                       retain_graph=True, create_graph=True)[0]

        img_real = x[0].permute(1, 2, 0)
        img_grad = grad_x[0].permute(1, 2, 0)
        #img_grad_b = grad_x_b[0].permute(1, 2, 0)
        img_grad_abs = (torch.abs(img_grad) - img_grad) #seems to give a decent pixel importance
        #img_grad_b_abs = torch.abs(img_grad_b) - img_grad_b

        # f, axarr = plt.subplots(2, 2)
        # axarr[0, 0].imshow(img_real.detach().numpy(), cmap='gray')
        # axarr[0, 1].imshow(img_grad.detach().numpy(), cmap='gray')
        # axarr[1, 0].imshow(img_grad_abs.detach().numpy(), cmap='gray')
        # axarr[1, 1].imshow(img_grad_abs.detach().numpy(), cmap='gray')
        # plt.show()
        # exit()

        #grad_a = torch.squeeze(grad_a)
        #grad_b = torch.squeeze(grad_b)

        #batch, c, h, w = grad_x.shape
        #grad_x = grad_x.view(c, h, w)

        ab, af, ah, aw = grad_a.shape
        bb, bf, bh, bw = grad_b.shape
        xb, xf, xh, xw = grad_x.shape


        #TODO: Set batch size here
        #keep separate for now, but can be combined eventually for memory savings
        final_grads_1 = torch.zeros((1, xf, xh, xw)) #layer 1
        final_grads_2 = torch.zeros((1, xf, xh, xw)) #layer 2

        filter_index = 0
        layer = 1
        for i in range(af):
            for j in range(ah):
                for k in range(aw):
                    scaled_grad = (self.pixel_importance_rank*self.scale_func(grad_a[0, i, j, k], filter_index+i, layer, y)) + \
                                  ((1-self.pixel_importance_rank)*self.scale_func(grad_a[0, i, j, k], filter_index+i, layer, y_targ))
                    s_grad = torch.autograd.grad(a[0, i, j, k], x,
                                retain_graph=True, create_graph=True, allow_unused=True)[0]

                    final_grads_1[0, :, :, :] += scaled_grad * s_grad[0, :, :, :]

        filter_index += af
        layer += 1
        for i in range(bf):
            for j in range(bh):
                for k in range(bw):
                    scaled_grad = (self.pixel_importance_rank*self.scale_func(grad_b[0, i, j, k], filter_index+i, layer, y)) + \
                                  ((1-self.pixel_importance_rank)*self.scale_func(grad_b[0, i, j, k], filter_index+i, layer, y_targ))
                    s_grad = torch.autograd.grad(b[0, i, j, k], x, 
                                retain_graph=True, create_graph=True, allow_unused=True)[0]

                    final_grads_2[0, :, :, :] += scaled_grad * s_grad[0, :, :, :]

        #scale grads by the pixel importance
        #final_grads_1[0, :, :, :] = final_grads_1[0, :, :, :] * self.pixel_importance
        #final_grads_2[0, :, :, :] = final_grads_2[0, :, :, :] * self.pixel_importance

        #for now take midpoint for return grad
        final_grads = (final_grads_1 + final_grads_2)/2
        return final_grads, cost

    def set_pixel_importance(self, images, labels):
        images.requires_grad = True

        loss = nn.CrossEntropyLoss()
        outputs, a, a_a, b, b_a = self.model.semi_forward(images)

        cost = loss(outputs[:1, :], labels[:1])

        grad_x = torch.autograd.grad(cost, images,
               retain_graph=True, create_graph=True)[0]

        #img_grad = grad_x[0].permute(1, 2, 0)
        img_grad = grad_x[0]
        img_grad_abs = (torch.abs(img_grad) - img_grad)

        self.pixel_importance[:, :, :] = img_grad_abs[:, :, :]

        f, h, w = self.pixel_importance.shape
        flat = f*h*w
        p_i = self.pixel_importance.view((flat))
        args = torch.argsort(p_i)
        p_i_g = torch.zeros((flat))
        for i in range(flat):
            p_i_g[args[i]] = i

        #transform rank into a decimal
        for i in range(flat):
            p_i_g[i] = 1 - (1/(flat))*p_i_g[i]

        p_i_g = p_i_g.view((f, h, w))

        self.pixel_importance_rank[:, :, :] = p_i_g[:, :, :]

        #NOTE: this seems to pass a checksum test when doign another grad, but may require more checking later
        #for now threshold to 0, 1, later, make it a ranked scaling

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

        self.set_pixel_importance(adv_images, labels)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            # outputs = self.model(adv_images)

            # # Calculate loss
            # if self._targeted:
            #     cost = -loss(outputs, target_labels)
            # else:
            #     cost = loss(outputs, labels)

            # # Update adversarial images
            # grad = torch.autograd.grad(cost, adv_images,
            #                            retain_graph=True, create_graph=True)[0]

            # print ('Pre-grad')
            # print (torch.sum(grad))

            #assume hardcoded for LeNet for now
            #a, a_i, b, b_i = self.calc_scaled_grads(adv_images, labels)

            # ratio_a = a/a_i
            # ratio_b = b/b_i

            # grad = ratio_a*ratio_b*grad

            # r1, r2 = self.calc_scaled_grads(adv_images, labels)

            # grad = (r1*grad + r2*grad) / 2

            #for now intercept here
            grad, cost = self.grad_play(adv_images, labels, target_labels)
            print (cost)

            # print ('Post-grad')
            # print (torch.sum(grad))

            # exit()

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            if (cost == 0):
                break

        return adv_images