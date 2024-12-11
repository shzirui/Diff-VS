import torch
import torch.nn.functional as F
import sys


class Noise_adap():

    def __init__(self, times):
        ceil = 0.0199
        prod = 0.9999
        while prod > 4.0358e-5:
            ceil += 0.0001
            betas_test = torch.linspace(0.0001, ceil, times)
            alphas_test = 1. - betas_test
            prod = torch.prod(alphas_test, axis=0)
        ceil = round(ceil, 4)
        self.betas = torch.linspace(0.0001, ceil, times)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        返回所传递的值列表vals中的特定索引，同时考虑到批处理维度。
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def add_noise(self, x, noise, t):
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def min_noise(self, x, noise, t):
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return (x - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def sub_noise(self, x, noise, t):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        gen = sqrt_recip_alphas_t * (x - betas_t * noise / sqrt_one_minus_alphas_cumprod_t)
        if t == 0:
            return gen
        else:
            noise = torch.randn_like(x)
            return gen + torch.sqrt(posterior_variance_t) * noise


class Noise():

    def __init__(self, times):
        self.betas = torch.linspace(0.0001, 0.02, times)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def get_index_from_list(self, vals, t, x_shape):
        """ 
        返回所传递的值列表vals中的特定索引，同时考虑到批处理维度。
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def add_noise(self, x, noise, t):
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def min_noise(self, x, noise, t):
        sqrt_alphas_cumprod_t = self.get_index_from_list(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        return (x - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def sub_noise(self, x, noise, t):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        gen = sqrt_recip_alphas_t * (x - betas_t * noise / sqrt_one_minus_alphas_cumprod_t)
        if t == 0:
            return gen
        else:
            noise = torch.randn_like(x)
            return gen + torch.sqrt(posterior_variance_t) * noise