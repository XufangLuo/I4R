import numpy as np
import math
import torch
from torch.autograd import Function


class MaxDivideMin(Function):

    @staticmethod
    def forward(ctx, input_rep, coef, count=None, f=None, logger=None):
        """
        calculate the condition number
        :param ctx:
        :param input_rep: (batch, dim_representation)
        :param coef: the coefficient for the gradients
        other params: for save sigmas to file
        :return:
        """
        ctx.save_for_backward(input_rep, torch.from_numpy(np.array([coef])).type(torch.FloatTensor))
        if count is not None:
            # full calculation
            u, s, v = torch.svd(input_rep, some=False)
            logger.write_sigma(f, count, s.numpy().tolist())
            number = s[0] / s[-1]
            return number
        else:
            # for fast
            return torch.sum(torch.zeros(1))

    @staticmethod
    def backward(ctx, grad_norm):
        input_rep, coef = ctx.saved_variables
        u, s, v = torch.svd(input_rep, some=True)  # to handle non-square matrix
        max_sigma = s[0]
        min_sigma = s[-1] if float(s[-1]) != 0 else (s[-1] + 0.00000000000000000001)
        grad_max = u[:, 0].view(-1,1).mm(v.t()[0, :].view(1, -1))
        grad_min = u[:, -1].view(-1,1).mm(v.t()[-1, :].view(1, -1))
        # calculate grad_input
        grad_input = min_sigma.expand_as(grad_max) * grad_max - max_sigma.expand_as(grad_min) * grad_min
        grad_input = (1/(min_sigma * min_sigma)).expand_as(grad_input) * grad_input
        # coef
        grad_input = coef.expand_as(grad_input) * grad_input
        # return u.mm(v.t()), None, None, None, None
        return grad_input, None, None, None, None


class MaxMinusMin(Function):

    @staticmethod
    def forward(ctx, input_rep, coef, count=None, f=None, logger=None):
        """
        calculate the condition number
        :param ctx:
        :param input_rep: (batch, dim_representation)
        :param coef: the coefficient for the gradients
        other params: for save sigmas to file
        :return:
        """
        ctx.save_for_backward(input_rep, torch.from_numpy(np.array([coef])).type(torch.FloatTensor))
        if count is not None:
            # full calculation
            u, s, v = torch.svd(input_rep, some=False)
            logger.write_sigma(f, count, s.numpy().tolist())
            number = s[0] - s[-1]
            return number
        else:
            # for fast
            return torch.sum(torch.zeros(1))

    @staticmethod
    def backward(ctx, grad_norm):
        input_rep, coef = ctx.saved_variables
        u, s, v = torch.svd(input_rep, some=True)  # to handle non-square matrix
        grad_max = u[:, 0].view(-1,1).mm(v.t()[0, :].view(1, -1))
        grad_min = u[:, -1].view(-1,1).mm(v.t()[-1, :].view(1, -1))
        # calculate grad_input
        grad_input = grad_max - grad_min
        # coef
        grad_input = coef.expand_as(grad_input) * grad_input
        # return u.mm(v.t()), None, None, None, None
        return grad_input, None, None, None, None
