'''
transformer模型
'''
import sys
import math
import torch
from torch import nn


# resnet->down block / up block
class Resnet(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.time = torch.nn.Sequential(
            torch.nn.SiLU(),
            torch.torch.nn.Linear(1280, dim_out),
            torch.nn.Unflatten(dim=1, unflattened_size=(dim_out, 1)),
        )

        self.s0 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_in,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv1d(dim_in,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.s1 = torch.nn.Sequential(
            torch.torch.nn.GroupNorm(num_groups=32,
                                     num_channels=dim_out,
                                     eps=1e-05,
                                     affine=True),
            torch.nn.SiLU(),
            torch.torch.nn.Conv1d(dim_out,
                                  dim_out,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1),
        )

        self.res = None
        if dim_in != dim_out:
            self.res = torch.torch.nn.Conv1d(dim_in,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

    def forward(self, x, time):
        # x -> [1, dim_in, 161]
        # time -> [1, 1280]

        res = x

        # [1, 1280] -> [1, dim_out, 1]
        time = self.time(time)

        # [1, dim_in, 161] -> [1, dim_out, 161]
        x = self.s0(x) + time

        # [1, dim_out, 161] -> [1, dim_out, 161]
        x = self.s1(x)

        # [1, dim_in, 161] -> [1, dim_out, 161]
        if self.res:
            res = self.res(res)

        # [1, dim_out, 161]
        x = res + x

        return x


# cross attention->transformer
class CrossAttention(torch.nn.Module):

    def __init__(self, dim_q, dim_kv):
        # dim_q -> 320
        # dim_kv -> 768

        super().__init__()

        self.dim_q = dim_q

        self.q = torch.nn.Linear(dim_q, dim_q, bias=False)
        self.k = torch.nn.Linear(dim_kv, dim_q, bias=False)
        self.v = torch.nn.Linear(dim_kv, dim_q, bias=False)

        self.out = torch.nn.Linear(dim_q, dim_q)

    def forward(self, q, kv):
        # x -> [1, 161, dim_q]
        # kv -> [1, 161, dim_kv]

        # [1, 161, dim_q] -> [1, 161, dim_q]
        q = self.q(q)
        # [1, 161, dim_kv] -> [1, 161, dim_q]
        k = self.k(kv)
        # [1, 161, dim_kv] -> [1, 161, dim_q]
        v = self.v(kv)

        def reshape(x):
            b, lens, dim = x.shape
            x = x.reshape(b, lens, 8, dim // 8)
            x = x.transpose(1, 2)
            x = x.reshape(b * 8, lens, dim // 8)

            return x

        # [1, 161, dim_q] -> [8, 161, dim_q//8]
        q = reshape(q)
        # [1, 161, dim_q] -> [8, 161, dim_q//8]
        k = reshape(k)
        # [1, 161, dim_q] -> [8, 161, dim_q//8]
        v = reshape(v)

        # [8, 161, dim_q//8] * [8, dim_q//8, 161] -> [8, 161, 161]
        atten = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device),
            q,
            k.transpose(1, 2),
            beta=0,
            alpha=(self.dim_q // 8)**-0.5,
        )

        atten = atten.softmax(dim=-1)

        # [8, 161, 161] * [8, 161, dim_q//8] -> [8, 161, dim_q//8]
        atten = atten.bmm(v)

        def reshape(x):
            # x -> [8, 4096, 40]
            b, lens, dim = x.shape

            # [8, 4096, 40] -> [1, 8, 4096, 40]
            x = x.reshape(b // 8, 8, lens, dim)

            # [1, 8, 4096, 40] -> [1, 4096, 8, 40]
            x = x.transpose(1, 2)

            # [1, 4096, 320]
            x = x.reshape(b // 8, lens, dim * 8)

            return x

        # [8, 161, dim_q//8] -> [1, 161, dim_q]
        atten = reshape(atten)

        # [1, 161, dim_q] -> [1, 161, dim_q]
        atten = self.out(atten)

        return atten


# transformer->down block / up block
class Transformer(torch.nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.dim = dim

        # in
        self.norm_in = torch.nn.GroupNorm(num_groups=32,
                                          num_channels=dim,
                                          eps=1e-6,
                                          affine=True)
        self.cnn_in = torch.nn.Conv1d(dim,
                                      dim,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)

        # atten
        self.norm_atten0 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten1 = CrossAttention(dim, dim)
        self.norm_atten1 = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.atten2 = CrossAttention(dim, 1024)

        # act
        self.norm_act = torch.nn.LayerNorm(dim, elementwise_affine=True)
        self.fc0 = torch.nn.Linear(dim, dim * 8)
        self.act = torch.nn.GELU()
        self.fc1 = torch.nn.Linear(dim * 4, dim)

        # out
        self.cnn_out = torch.nn.Conv1d(dim,
                                       dim,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)

    def forward(self, q, kv):
        # q -> [1, dim, 161]
        # kv -> [1, 161, 1024]
        b, _, l = q.shape
        res1 = q

        # ----in----
        # [1, dim, 161] -> [1, dim, 161]
        q = self.cnn_in(self.norm_in(q))

        # [1, dim, 161] -> [1, 161, dim]
        q = q.permute(0, 2, 1).reshape(b, l, self.dim)

        # ----atten----
        # [1, 161, dim] -> [1, 161, dim]
        q = self.atten1(q=self.norm_atten0(q), kv=self.norm_atten0(q)) + q
        # [1, 161, dim] -> [1, 161, dim]
        q = self.atten2(q=self.norm_atten1(q), kv=kv) + q

        # ----act----
        # [1, 161, dim]
        res2 = q

        # [1, 161, dim] -> [1, 161, dim*8]
        q = self.fc0(self.norm_act(q))

        # dim*4
        d = q.shape[2] // 2

        # [1, 161, dim*4] * [1, 161, dim*4] -> [1, 161, dim*4]
        q = q[:, :, :d] * self.act(q[:, :, d:])

        # [1, 161, dim*4] -> [1, 161, dim]
        q = self.fc1(q) + res2

        #----out----
        # [1, 161, dim] -> [1, dim, 161]
        q = q.reshape(b, l, self.dim).permute(0, 2, 1).contiguous()

        # [1, dim, 161]
        q = self.cnn_out(q) + res1

        return q


# down block->unet
class DownBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.tf0 = Transformer(dim_out)
        self.res0 = Resnet(dim_in, dim_out)

        self.tf1 = Transformer(dim_out)
        self.res1 = Resnet(dim_out, dim_out)

        self.out = torch.nn.Conv1d(dim_out,
                                   dim_out,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, out_vae, out_encoder, time):
        outs = []

        # [1, dim_in, 161] -> [1, dim_out, 161]
        out_vae = self.res0(out_vae, time)
        # [1, dim_out, 161] -> [1, dim_out, 161]
        out_vae = self.tf0(out_vae, out_encoder)
        # [1, dim_out, 161]
        outs.append(out_vae)

        # [1, dim_out, 161] -> [1, dim_out, 161]
        out_vae = self.res1(out_vae, time)
        # [1, dim_out, 161] -> [1, dim_out, 161]
        out_vae = self.tf1(out_vae, out_encoder)
        # [1, dim_out, 161], [1, dim_out, 161]
        outs.append(out_vae)

        # [1, dim_out, 161] -> [1, dim_out, 161]
        out_vae = self.out(out_vae)
        # [1, dim_out, 161], [1, dim_out, 161], [1, dim_out, 161]
        outs.append(out_vae)

        return out_vae, outs


# up block->unet
class UpBlock(torch.nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.res0 = Resnet(dim_in + dim_in, dim_out)
        self.res1 = Resnet(dim_out + dim_in, dim_out)
        self.res2 = Resnet(dim_out + dim_in, dim_out)

        self.tf0 = Transformer(dim_out)
        self.tf1 = Transformer(dim_out)
        self.tf2 = Transformer(dim_out)
        
        self.out = torch.nn.Conv1d(dim_out, dim_out, kernel_size=3, padding=1, stride=1)

    def forward(self, out_vae, out_encoder, time, out_down):
        # [1, dim_in, 161], [1, dim_in, 161] -> [1, dim_out, 161]
        out_vae = self.res0(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf0(out_vae, out_encoder)

        # [1, dim_out, 161], [1, dim_in, 161] -> [1, dim_out, 161]
        out_vae = self.res1(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf1(out_vae, out_encoder)

        # [1, dim_out, 161], [1, dim_in, 161] -> [1, dim_out, 161]
        out_vae = self.res2(torch.cat([out_vae, out_down.pop()], dim=1), time)
        out_vae = self.tf2(out_vae, out_encoder)

        # [1, dim_out, 161] -> [1, dim_out, 161]
        out_vae = self.out(out_vae)

        return out_vae


class MLP(torch.nn.Module):

    def __init__(self, feature_dim=1024):
        super(MLP, self).__init__()
        self.feature_dim = feature_dim
        
        # 1D Convolution with kernel size 3
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=3, padding=1)
        
        # Global Average Pooling followed by a Conv1d with kernel size 1 to simulate a conv with kernel size L
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv2 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=1)
        
        # Fully connected layer
        self.fc = nn.Linear(feature_dim * 3, feature_dim)
        
    def forward(self, x):
        # x shape: (batch_size, feature_dim, L)
        
        # Apply first convolution
        out1 = self.conv1(x)
        
        # Apply global average pooling and the second convolution
        out2 = self.global_avg_pool(x)  # Shape: (batch_size, feature_dim, 1)
        out2 = self.conv2(out2)  # Shape: (batch_size, feature_dim, 1)
        out2 = out2.expand(-1, -1, x.size(2))  # Expand to (batch_size, feature_dim, L)
        
        # Concatenate along the feature dimension
        out = torch.cat((x, out1, out2), dim=1)  # Shape: (batch_size, feature_dim * 3, L)
        
        # Apply fully connected layer
        out = out.permute(0, 2, 1)  # Shape: (batch_size, L, feature_dim * 3)
        out = self.fc(out)  # Shape: (batch_size, L, feature_dim)
        
        out = out.permute(0, 2, 1)  # Shape: (batch_size, feature_dim, L)
        
        return out


# unet模型：1层in, 4层down, 1层middle, 4层up, 1层out
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.mlp = MLP(1024)

        # in
        self.in_vae = torch.nn.Conv1d(1, 320, kernel_size=3, padding=1)

        self.in_time = torch.nn.Sequential(
            torch.nn.Linear(320, 1280),
            torch.nn.SiLU(),
            torch.nn.Linear(1280, 1280),
        )

        # down
        self.down_block0 = DownBlock(320, 640)
        self.down_block1 = DownBlock(640, 1280)

        # mid
        self.mid_res0 = Resnet(1280, 1280)
        self.mid_tf = Transformer(1280)
        self.mid_res1 = Resnet(1280, 1280)

        # up
        self.up_block0 = UpBlock(1280, 640)
        self.up_block1 = UpBlock(640, 320)

        # out
        self.out = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=320, num_groups=32, eps=1e-5),
            torch.nn.SiLU(),
            torch.nn.Conv1d(320, 1, kernel_size=3, padding=1),
        )

    def forward(self, out_vae, out_encoder, time):
        # out_vae -> [1, 161, 1]
        # out_encoder -> [1, 161, 1024]
        # time -> [1]

        # output_encoder = self.mlp(out_encoder)

        # ----in----
        # [1, 161, 1] -> [1, 320, 161]
        out_vae = self.in_vae(out_vae.permute(0,2,1))

        def get_time_embed(t):
            # -9.210340371976184 = -math.log(10000)
            e = torch.arange(160) * -9.210340371976184 / 160
            e = e.exp().to(t.device) * t

            # [160+160] -> [320] -> [1, 320]
            e = torch.cat([e.cos(), e.sin()]).unsqueeze(dim=0)

            return e

        # [1] -> [1, 320]
        time = get_time_embed(time)
        # [1, 320] -> [1, 1280]
        time = self.in_time(time)

        # ----down----
        out_down = [out_vae]

        # [1, 320, 161], [1, 161, 1024], [1, 1280] -> [1, 640, 161]
        # out -> [1, 640, 161], [1, 640, 161], [1, 640, 161]
        out_vae, out = self.down_block0(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        # [1, 640, 161], [1, 161, 1024], [1, 1280] -> [1, 1280, 161]
        # out -> [1, 1280, 161], [1, 1280, 161], [1, 1280, 161]
        out_vae, out = self.down_block1(out_vae=out_vae,
                                        out_encoder=out_encoder,
                                        time=time)
        out_down.extend(out)

        # ----mid----
        # [1, 1280, 161], [1, 1280] -> [1, 1280, 161]
        out_vae = self.mid_res0(out_vae, time)

        # [1, 1280, 161], [1, 161, 1024] -> [1, 1280, 161]
        out_vae = self.mid_tf(out_vae, out_encoder)

        # [1, 1280, 161], [1, 1280] -> [1, 1280, 161]
        out_vae = self.mid_res1(out_vae, time)

        # ----up----
        # [1, 1280, 161], [1, 161, 1024], [1, 1280] -> [1, 640, 161]
        # out_down -> [1, 1280, 161],[1, 1280, 161],[1, 1280, 161]
        out_vae = self.up_block0(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        # [1, 640, 161], [1, 161, 1024], [1, 1280] -> [1, 320, 161]
        # out_down -> [1, 640, 161],[1, 640, 161],[1, 640, 161]
        out_vae = self.up_block1(out_vae=out_vae,
                                 out_encoder=out_encoder,
                                 time=time,
                                 out_down=out_down)

        # ----out----
        # [1, 320, 161] -> [1, 161, 1]
        out_vae = self.out(out_vae).permute(0,2,1)

        return out_vae