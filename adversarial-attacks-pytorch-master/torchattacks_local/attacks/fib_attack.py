import torch
import torch.nn as nn

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

    def fi_norm(self, val):
        return (val**2)

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

            grad_a = torch.squeeze(grad_a)
            grad_b = torch.squeeze(grad_b)

            f_index = 0
            f, h, w = grad_a.shape
            for i in range(f_index, f):
                for j in range(h):
                    for k in range(w):
                        self.fi_dict[i] += self.fi_norm(grad_a[i, j, k])

            f_index += f
            f, h, w = grad_b.shape
            for i in range(f_index, f):
                for j in range(h):
                    for k in range(w):
                        self.fi_dict[i] += self.fi_norm(grad_b[i, j, k])

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

    def accumulation_func(self, grad):
        return (torch.sum(grad))

    def propagated_grad(self, grad_data, grad_conv, f_index_ref):
        f, h, w = grad_data.shape
        grad_ret = torch.zeros((f, h, w))
        grad_ret_i = torch.zeros((f, h, w))

        f_index_start = f_index_ref
        f1, h1, w1 = grad_data.shape
        f2, h2, w2 = grad_conv.shape
        for i in range(f1):
            for j in range(h1):
                for k in range(w1):
                    accumulation = 0
                    accumulation_i = 0
                    f_index = f_index_start
                    for ii in range(f2):
                        for jj in range(h2):
                            for kk in range(w2):
                                scaling = self.scale_func(grad_conv[ii, jj, kk], f_index)
                                partial = (grad_data[i, j, k] / grad_conv[ii, jj, kk]) #TODO: div 0 issue
                                accumulation += (scaling*partial)
                                accumulation_i += (grad_conv[ii, jj, kk]*partial)
                        f_index += 1
                    grad_ret[i, j, k] = accumulation
                    grad_ret_i[i, j, k] = accumulation_i

        return grad_ret, grad_ret_i

    def non_propagated_grad(self, grad_conv, f_index_ref):
        f_index_start = f_index_ref
        f1, h1, w1 = grad_conv.shape

        accumulation = 0
        accumulation_i = 0
        f_index = f_index_start
        for i in range(f1):
            for j in range(h1):
                for k in range(w1):
                    scaling = self.scale_func(grad_conv[i, j, k], f_index)
                    accumulation += (scaling)
                    accumulation_i += grad_conv[i, j, k]
            f_index += 1

        grad_ratio = 0
        if (accumulation_i != 0):
            grad_ratio = accumulation/accumulation_i

        return grad_ratio

    def replicate_input_withgrad(self, x):
        return x.detach().clone().requires_grad_()

    def jacobian(self, model, x, output_class):
        """
        Compute the output_class'th row of a Jacobian matrix. In other words,
        compute the gradient wrt to the output_class.
        :param model: forward pass function.
        :param x: input tensor.
        :param output_class: the output class we want to compute the gradients.
        :return: output_class'th row of the Jacobian matrix wrt x.
        """
        xvar = self.replicate_input_withgrad(x)
        scores = model(xvar)

        # compute gradients for the class output_class wrt the input x
        # using backpropagation
        torch.sum(scores[:, output_class]).backward()

        return xvar.grad.detach().clone()

    def compute_forward_derivative(self, xadv, y):
        jacobians = torch.stack([self.jacobian(self.model, xadv, yadv)
                                 for yadv in range(self.num_classes)])
        grads = jacobians.view((jacobians.shape[0], jacobians.shape[1], -1))
        grads_target = grads[y, range(len(y)), :]
        grads_other = grads.sum(dim=0) - grads_target
        return grads_target, grads_other

    def calc_scaled_grads(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        data.requires_grad = True

        loss = nn.CrossEntropyLoss()

        if self._targeted:
            target_labels = self._get_target_label(data, target)

        target_labels = target_labels.to(self.device)

        #lets try some stuff out here
        v = self.grad_play(data, target, target_labels)

        outputs, a, a_a, b, b_a = self.model.semi_forward(data)

        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, target)

        grad_a = torch.autograd.grad(cost, a,
                       retain_graph=True, create_graph=True)[0]

        grad_a_a = torch.autograd.grad(cost, a_a,
                       retain_graph=True, create_graph=True)[0]


        grad_b = torch.autograd.grad(cost, b,
                       retain_graph=True, create_graph=True)[0]

        grad_b_a = torch.autograd.grad(cost, b_a,
                       retain_graph=True, create_graph=True)[0]

        grad_x = torch.autograd.grad(cost, data,
                       retain_graph=True, create_graph=True)[0]

        grad_a = torch.squeeze(grad_a)
        grad_a_a = torch.squeeze(grad_a_a)
        grad_b = torch.squeeze(grad_b)
        grad_b_a = torch.squeeze(grad_b_a)

        batch, c, h, w = grad_x.shape
        grad_x = grad_x.view(c, h, w)

        # grad_first, grad_first_i = self.propagated_grad(grad_x, grad_a, 0)
        # grad_second, grad_second_i = self.propagated_grad(grad_a_a, grad_b, 10)

        # grad_first_value = self.accumulation_func(grad_first)
        # grad_first_value_i = self.accumulation_func(grad_first_i)
        # grad_second_value = self.accumulation_func(grad_second)
        # grad_second_value_i = self.accumulation_func(grad_second_i)

        #return grad_first_value, grad_first_value_i, grad_second_value, grad_second_value_i

        r1 = self.non_propagated_grad(grad_a, 0)
        r2 = self.non_propagated_grad(grad_b, 10)

        return r1, r2

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

        if (rank > filter_range_low and rank < filter_range_high):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_step_high(self, val, imp, rank, alpha):
        scaled_val = 0

        filter_range_low = 28
        filter_range_high = 30

        if (rank > filter_range_low and rank < filter_range_high):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_step_low(self, val, imp, rank, alpha):
        scaled_val = 0

        filter_range_low = 0
        filter_range_high = 2

        if (rank > filter_range_low and rank < filter_range_high):
            scaled_val = alpha*val
        else:
            scaled_val = 0

        return scaled_val

    def scale_gaussian(self, val, imp, alpha):
        #TODO: still to implement
        return val

    def scale_func(self, grad, f_index):
        sgn = torch.sign(grad)
        val = torch.abs(grad)
        imp = self.fi_dict[f_index]
        rank = self.fi_dict_rankings[f_index]

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

        return (sgn*scaled_val)

    def grad_play(self, x, y, y_targ):
        x, y = x.to(self.device), y.to(self.device)
        x.requires_grad = True
        loss = nn.CrossEntropyLoss()

        outputs, a, a_a, b, b_a = self.model.semi_forward(x)
        cost = -loss(outputs, y_targ)

        grad_x = torch.autograd.grad(cost, x,
                       retain_graph=True, create_graph=True)[0]

        grad_a = torch.autograd.grad(cost, a,
                       retain_graph=True, create_graph=True)[0]

        grad_a = torch.squeeze(grad_a)

        batch, c, h, w = grad_x.shape
        grad_x = grad_x.view(c, h, w)

        f, h, w = grad_a.shape
        xf, xh, xw = grad_x.shape

        final_grads = torch.zeros((1, xf, xh, xw)) #TODO: Set batch size here

        print ('Starting Grad Accum')
        grad_acum = 0
        s_grad_acum = 0
        for i in range(f):
            for j in range(h):
                for k in range(w):
                    scaled_grad = self.scale_func(grad_a[i, j, k], i)
                    s_grad = torch.autograd.grad(a[0, i, j, k], x,
                                retain_graph=True, create_graph=True, allow_unused=True)[0]


                    final_grads[0, :, :, :] += scaled_grad * s_grad[0, :, :, :]
        return final_grads

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

        loss = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=True, create_graph=True)[0]

            print ('Pre-grad')
            print (torch.sum(grad))

            #assume hardcoded for LeNet for now
            #a, a_i, b, b_i = self.calc_scaled_grads(adv_images, labels)

            # ratio_a = a/a_i
            # ratio_b = b/b_i

            # grad = ratio_a*ratio_b*grad

            # r1, r2 = self.calc_scaled_grads(adv_images, labels)

            # grad = (r1*grad + r2*grad) / 2

            #for now intercept here
            grad = self.grad_play(adv_images, labels, target_labels)

            print ('Post-grad')
            print (torch.sum(grad))

            # exit()

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images