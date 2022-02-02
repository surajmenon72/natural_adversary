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

    def __init__(self, model, data_loader, eps=8/255, alpha=2/255, steps=5, decay=1.0):
        super().__init__("FIBA", model)
        self.eps = eps
        self.steps = steps
        self.decay = decay
        self.alpha = alpha
        self._supported_mode = ['default', 'targeted']
        self.data_loader = data_loader
        self.total_filters = (model.conv1.out_channels+model.conv2.out_channels) #needs to be done automated
        self.fi_dict = torch.zeros(self.total_filters)

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
            print (iteration)
            if (iteration >= sample_thresh):
                break

        #once all accumulation is done, now we can normalize
        total = torch.sum(self.fi_dict, axis=0)
        for i in range(self.total_filters):
            self.fi_dict[i] /= total

        print ('Set Filter Importance Dict')
        print (self.fi_dict)

    def scale_identity(self, val, imp):
        return val

    def scale_linear(self, val, imp, alpha):
        return alpha*imp*val

    def scale_inverse(self, val, imp, alpha):
        return alpha*(1-imp)*val

    def scale_gaussian(self, val, imp, alpha):
        #still to implement
        return val

    def scale_func(self, grad, f_index):
        sgn = torch.sign(grad)
        val = torch.abs(grad)
        imp = self.fi_dict[f_index]

        #select scaling func, make set in main func
        scaled_val = self.scale_identity(val, imp)
        #scaled_val = self.scale_inverse(val, imp, 1)

        return (sgn*scaled_val)

    def accumulation_func(self, grad):
        return (torch.sum(grad))

    def calc_scaled_grads(self, data, target):
        data, target = data.to(self.device), target.to(self.device)
        data.requires_grad = True

        loss = nn.CrossEntropyLoss()

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

        
        print ('Accumulated Grads') #abstract to func, try to vectorize
        print (grad_a.shape)
        print (grad_a_a.shape)
        print (grad_b.shape)
        print (grad_b_a.shape)
        print (grad_x.shape)
        
        f, h, w = grad_a_a.shape
        grad_second = torch.zeros((f, h, w))
        grad_second_i = torch.zeros((f, h, w))
        f, h, w = grad_x.shape
        grad_first = torch.zeros((f, h, w))
        grad_first_i = torch.zeros((f, h, w))

        f_index_start = 0
        f1, h1, w1 = grad_x.shape
        f2, h2, w2 = grad_a.shape
        #for 1st conv layer
        for i in range(f1):
            for j in range(h1):
                for k in range(w1):
                    accumulation = 0
                    accumulation_i = 0
                    f_index = f_index_start
                    for ii in range(f2):
                        for jj in range(h2):
                            for kk in range(w2):
                                scaling = self.scale_func(grad_a[ii, jj, kk], f_index)
                                partial = (grad_x[i, j, k] / grad_a[ii, jj, kk])
                                accumulation += (scaling*partial)
                                accumulation_i += (grad_a[ii, jj, kk]*partial)
                        f_index += 1
                    grad_first[i, j, k] = accumulation
                    grad_first_i[i, j, k] = accumulation_i

        f_index_start = 10
        f1, h1, w1 = grad_a_a.shape
        f2, h2, w2 = grad_b.shape
        #first for 2nd conv layer, propagate through 1st activation
        for i in range(f1):
            for j in range(h1):
                for k in range(w1):
                    accumulation = 0
                    accumulation_i = 0
                    f_index = f_index_start
                    for ii in range(f_index, f2):
                        for jj in range(h2):
                            for kk in range(w2):
                                scaling = self.scale_func(grad_b[ii, jj, kk], f_index)
                                partial = (grad_a_a[i, j, k] / grad_b[ii, jj, kk])
                                accumulation += (scaling*partial)
                                accumulation_i += (grad_b[ii, jj, kk]*partial)
                        f_index += 1
                    grad_second[i, j, k] = accumulation
                    grad_second_i[i, j, k] = accumulation_i

        grad_first_value = self.accumulation_func(grad_first)
        grad_first_value_i = self.accumulation_func(grad_first_i)
        grad_second_value = self.accumulation_func(grad_second)
        grad_second_value_i = self.accumulation_func(grad_second_i)

        return grad_first_value, grad_first_value_i, grad_second_value, grad_second_value_i

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

            #assume hardcoded for LeNet for now
            a, a_i, b, b_i = self.calc_scaled_grads(adv_images, labels)

            print (a)
            print (a_i)
            print (b)
            print (b_i)
            exit()

            # ratio_a = a/a_i
            # ratio_b = b/b_i

            # grad = ratio_a*ratio_b*grad

            grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() + self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images