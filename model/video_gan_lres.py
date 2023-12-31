# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import dnnlib
import einops
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_utils.distributed as dist_utils
import utils
from torch.optim import Adam
from torch_utils import misc, training_stats
from torchvision import transforms,models
from PIL import Image
import numpy as np

import clip

from model.diff_augment import DiffAugment

# =====================================================================================================================


@dataclass
class LowResVideoGAN:
    seq_length: int
    height: int
    width: int
    channels: int = 3

    G_lrate: float = 0.003
    G_beta2: float = 0.99
    G_warmup_steps: int = 0
    G_ema_beta: float = 0.99985
    G_ema_warmup_steps: int = 25000
    G_magnitude_ema_beta: float = 0.999
    G_grad_accum: int = 1
    G_kwargs: dict[str, Any] = field(default_factory=dict)
    G_random_temp_translate: bool = False

    D_lrate: float = 0.002
    D_beta2: float = 0.99
    D_warmup_steps: int = 0
    D_grad_accum: int = 1
    D_kwargs: dict[str, Any] = field(default_factory=dict)
    r1_gamma: Optional[float] = 10.0

    temp_scale_augment: float = 0.0
    diffaug_policy: str = "color,translation,cutout"

    # ==================================================================================================================

    def __post_init__(self):

        # Constructs generator and exponential moving average of generator.
        G_kwargs = dict(out_height=self.height, out_width=self.width)
        self.G = dnnlib.util.construct_class_by_name(**G_kwargs, **self.G_kwargs)
        self.G_ema = dnnlib.util.construct_class_by_name(**G_kwargs, **self.G_kwargs)
        self.G.cuda().requires_grad_(False).train()
        self.G_ema.cuda().requires_grad_(False).eval()

        # self.clip_model, self.preprocess = clip.load('ViT-B/32', "cuda")
        # self.clip_model.requires_grad_(False)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.vgg_model = models.vgg16(pretrained=True).features[:9]
        self.vgg_model.eval()
        self.vgg_model.requires_grad_(False)
        self.vgg_model.to("cuda")

        self.D = dnnlib.util.construct_class_by_name(
            seq_length=self.seq_length,
            max_edge=max(self.height, self.width),
            **self.D_kwargs,
        )
        self.D.cuda().requires_grad_(False).train()

        # Synchronizes all weights at start.
        for network in (self.G, self.D):
            for tensor in misc.params_and_buffers(network):
                dist.broadcast(tensor, src=0)

        for tensor, tensor_ema in zip(misc.params_and_buffers(self.G), misc.params_and_buffers(self.G_ema)):
            tensor_ema.copy_(tensor)

        # Constructs opts.
        self.G_opt = Adam(self.G.parameters(), lr=self.G_lrate, betas=(0, self.G_beta2))
        self.D_opt = Adam(self.D.parameters(), lr=self.D_lrate, betas=(0, self.D_beta2))

    # ==================================================================================================================

    def update_lrates(self, step: int):
        G_lrate = self.G_lrate * min((step + 1) / (self.G_warmup_steps + 1), 1.0)
        D_lrate = self.D_lrate * min((step + 1) / (self.D_warmup_steps + 1), 1.0)
        self.G_opt.param_groups[0]["lr"] = G_lrate
        self.D_opt.param_groups[0]["lr"] = D_lrate
        training_stats.report0(f"progress/G_lrate", G_lrate)
        training_stats.report0(f"progress/D_lrate", D_lrate)

    # ==================================================================================================================

    def update_G(self, batch: int,real_video: torch.Tensor):
        misc.assert_shape(real_video, (None, self.channels, self.seq_length, self.height, self.width))
        assert batch % self.G_grad_accum == 0
        assert batch // self.G_grad_accum >= 1

        self.G.requires_grad_(True)

        for _ in range(self.G_grad_accum):
            video = self.G(
                batch // self.G_grad_accum,
                self.seq_length + int(self.G_random_temp_translate) * self.G.total_temporal_scale,
                real_video[:,:,0,:,:],
                real_video[:,:,self.seq_length-1,:,:],
            )

            if self.G_random_temp_translate: #* False
                batch_size = video.size(0)
                t0 = torch.randint(video.size(2) - self.seq_length, (batch_size,))
                t1 = t0 + self.seq_length
                video = torch.stack([video[i, :, t0[i] : t1[i]] for i in range(batch_size)])

            logits = self.run_D(video)
            logits_loss = F.softplus(-logits)

            # begin_sim_loss, end_sim_loss, between_sim_loss = self.clip_loss(video,real_video)
            begin_sim_loss, end_sim_loss, between_sim_loss = self.perceptual_loss(video,real_video)
            L1_loss = F.l1_loss(video[:,:,0,:,:],real_video[:,:,0,:,:])+F.l1_loss(video[:,:,self.seq_length-1,:,:],real_video[:,:,self.seq_length-1,:,:])
            
            loss = logits_loss + begin_sim_loss + end_sim_loss + between_sim_loss + L1_loss
            loss = logits_loss

            loss.mean().backward()

            training_stats.report(f"loss/G_score", logits)
            training_stats.report(f"loss/G_sign", logits.sign())
            training_stats.report(f"loss/G_begin_sim", begin_sim_loss)
            training_stats.report(f"loss/G_end_sim", end_sim_loss)
            training_stats.report(f"loss/G_between_sim", between_sim_loss)
            training_stats.report(f"loss/G_reconstruct", L1_loss)
            training_stats.report(f"loss/G_loss", loss)

        self.G.requires_grad_(False)

        gain = 1 / self.G_grad_accum
        utils.sync_grads(self.G, gain=gain)
        self.G_opt.step()
        self.G_opt.zero_grad(set_to_none=True)

        return loss.mean()

    # ==================================================================================================================

    def update_D(self, real_video: torch.Tensor):
        misc.assert_shape(real_video, (None, self.channels, self.seq_length, self.height, self.width))
        assert real_video.size(0) % self.D_grad_accum == 0
        assert real_video.size(0) // self.D_grad_accum >= 1

        fake_video = self.G(
            real_video.size(0),
            self.seq_length + int(self.G_random_temp_translate) * self.G.total_temporal_scale,
            real_video[:,:,0,:,:],
            real_video[:,:,self.seq_length-1,:,:],
            magnitude_ema_beta=self.G_magnitude_ema_beta,
        )

        if self.G_random_temp_translate:
            batch_size = fake_video.size(0)
            t0 = torch.randint(fake_video.size(2) - self.seq_length, (batch_size,))
            t1 = t0 + self.seq_length
            fake_video = torch.stack([fake_video[i, :, t0[i] : t1[i]] for i in range(batch_size)])

        self.D.requires_grad_(True)

        for fake_video_chunk, real_video_chunk in zip(
            fake_video.chunk(self.D_grad_accum), real_video.chunk(self.D_grad_accum)
        ):
            fake_logits = self.run_D(fake_video_chunk)
            fake_loss = F.softplus(fake_logits)
            fake_loss.mean().backward()

            real_logits = self.run_D(real_video_chunk)
            real_loss = F.softplus(-real_logits)
            real_loss.mean().backward()

            training_stats.report(f"loss/D_score_fake", fake_logits)
            training_stats.report(f"loss/D_score_real", real_logits)
            training_stats.report(f"loss/D_sign_fake", fake_logits.sign())
            training_stats.report(f"loss/D_sign_real", real_logits.sign())
            training_stats.report(f"loss/D_loss", fake_loss + real_loss)

        self.D.requires_grad_(False)

        gain = 1 / self.D_grad_accum
        utils.sync_grads(self.D, gain=gain)
        self.D_opt.step()
        self.D_opt.zero_grad(set_to_none=True)

        return real_loss.mean()

    # ==================================================================================================================

    def update_r1(self, video: torch.Tensor, gain: float = 1.0):
        misc.assert_shape(video, (None, self.channels, self.seq_length, self.height, self.width))
        assert video.size(0) % self.D_grad_accum == 0
        assert video.size(0) // self.D_grad_accum >= 1

        self.D.requires_grad_(True)

        for video_chunk in video.chunk(self.D_grad_accum):
            video_chunk = video_chunk.detach().requires_grad_(True)
            logits = self.run_D(video_chunk)
            r1_grads = torch.autograd.grad(outputs=[logits.sum()], inputs=[video_chunk], create_graph=True)

            r1_penalty = r1_grads[0].square().sum(dim=(1, 2, 3, 4))
            r1_loss = r1_penalty * (self.r1_gamma / 2)
            r1_loss.mean().backward()

            training_stats.report(f"loss/r1_penalty", r1_penalty)
            training_stats.report(f"loss/r1_loss", r1_loss)

        self.D.requires_grad_(False)

        gain /= self.D_grad_accum
        utils.sync_grads(self.D, gain=gain)
        self.D_opt.step()
        self.D_opt.zero_grad(set_to_none=True)

    # ==================================================================================================================

    @torch.no_grad()
    def update_G_ema(self, step: int):
        ema_reciprocal_halflife = math.log(self.G_ema_beta, 0.5) * (self.G_ema_warmup_steps + 1) / (step + 1)
        ema_beta = min(0.5**ema_reciprocal_halflife, self.G_ema_beta)
        training_stats.report0("progress/G_ema_beta", ema_beta)
        for tensor_ema, tensor in zip(misc.params_and_buffers(self.G_ema), misc.params_and_buffers(self.G)):
            tensor_ema.lerp_(tensor, 1.0 - ema_beta)

    # ==================================================================================================================

    def ckpt(self):
        train_ckpt = dict(
            G_opt=self.G_opt.state_dict(),
            D_opt=self.D_opt.state_dict(),
        )
        for name, module in (("G_ema", self.G_ema), ("G", self.G), ("D", self.D)):
            if module is None:
                train_ckpt[name] = None
            else:
                module = copy.deepcopy(module).eval().requires_grad_(False)
                if dist_utils.get_world_size() > 1:
                    misc.check_ddp_consistency(module)
                train_ckpt[name] = module

        G_ema_ckpt = train_ckpt.pop("G_ema")
        return G_ema_ckpt, train_ckpt

    # ==================================================================================================================

    def run_D(self, video: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(video, (None, self.channels, self.seq_length, self.height, self.width))

        video = DiffAugment(video, self.diffaug_policy)

        if self.temp_scale_augment > 0:
            video = einops.rearrange(video, "n c t h w -> n c h w t")
            vs = []
            for v in video:
                scale_factor = 2 ** torch.empty(()).uniform_(-self.temp_scale_augment, self.temp_scale_augment)
                v = F.interpolate(
                    v,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                    scale_factor=(1, scale_factor),
                )
                p0 = torch.randint(max(0, self.seq_length - v.size(-1)) + 1, ()).item()
                p1 = max(0, self.seq_length - v.size(-1) - p0)
                v = F.pad(v, (p0, p1))

                i0 = torch.randint(v.size(-1) - self.seq_length + 1, ()).item()
                i1 = i0 + self.seq_length
                v = v[..., i0:i1]
                vs.append(v)
            video = torch.stack(vs)
            video = einops.rearrange(video, "n c h w t -> n c t h w")

        logits = self.D(video)
        return logits
    
    def perceptual_loss(self, video: torch.Tensor, real_video: torch.Tensor):
        misc.assert_shape(video, (None, self.channels, self.seq_length, self.height, self.width))

        batch_num = video.shape[0]

        begin_sim_loss = 0
        end_sim_loss = 0
        between_sim_loss = 0

        # for b in range(batch_num):
    
        #* 生成影片的clip feature
        image_features_begin = []
        for t in range(20):
            image_t = video[:, :, t, :, :]
            image_feature = self.vgg_model(self.normalize(image_t))
            image_features_begin.append(image_feature)

        image_features_end = []
        for t in range(20):
            image_t = video[:, :, self.seq_length-1-t, :, :]
            image_feature = self.vgg_model(self.normalize(image_t))
            image_features_end.append(image_feature)

        #* 真實第一張 clip feature
        real_img_begin = real_video[:,:,0,:,:]
        real_begin_feature = self.vgg_model(self.normalize(real_img_begin))

        #* 真實最後一張 clip feature
        real_img_end = real_video[:,:,self.seq_length-1,:,:]
        real_end_feature = self.vgg_model(self.normalize(real_img_end))

        #* [1,0.95,0.9,0.85, ...]
        similar_weight = np.arange(1, 1 - 0.05 * self.seq_length, -0.05)
        similar_weight[similar_weight<0] = 0
        weight_sum =  np.sum(similar_weight)

        #* clip image feature 的 cosine similarity loss
        for i in range(20):
            begin_sim_loss += similar_weight[i]*F.mse_loss(real_begin_feature,image_features_begin[i],reduction="mean")
            end_sim_loss += similar_weight[i]*F.mse_loss(real_end_feature,image_features_end[i],reduction="mean")
            if i<19:
                between_sim_loss += F.mse_loss(image_features_begin[i],image_features_begin[i+1],reduction="mean")
                between_sim_loss += F.mse_loss(image_features_end[i],image_features_end[i+1],reduction="mean")

        #* normalize 回cosine 範圍 (-1,1)
        begin_sim_loss /= weight_sum
        end_sim_loss /= weight_sum
        between_sim_loss /= (20*2-2)
        
        # begin_sim_loss /= batch_num
        # end_sim_loss /= batch_num
        # between_sim_loss /= batch_num

        #* 1-cosine => 越相近loss 越小
        return begin_sim_loss,end_sim_loss,between_sim_loss
'''
    def clip_loss(self, video: torch.Tensor, real_video: torch.Tensor):
        misc.assert_shape(video, (None, self.channels, self.seq_length, self.height, self.width))

        batch_num = video.shape[0]

        begin_sim_loss = 0
        end_sim_loss = 0
        between_sim_loss = 0

        to_pil = transforms.ToPILImage()

        for b in range(batch_num):
        
            #* 生成影片的clip feature
            image_features_begin = []
            for t in range(20):
                image_t = video[b, :, t, :, :]
                image = to_pil(image_t)
                image = self.preprocess(image).unsqueeze(0).to("cuda")
                image_feature = self.clip_model.encode_image(image)
                image_features_begin.append(image_feature)

            image_features_end = []
            for t in range(20):
                image_t = video[b, :, self.seq_length-1-t, :, :]
                image = to_pil(image_t)
                image = self.preprocess(image).unsqueeze(0).to("cuda")
                image_feature = self.clip_model.encode_image(image)
                image_features_end.append(image_feature)

            #* 真實第一張 clip feature
            real_img_begin = real_video[b,:,0,:,:]
            image = to_pil(real_img_begin)
            image = self.preprocess(image).unsqueeze(0).to("cuda")
            real_begin_feature = self.clip_model.encode_image(image)

            #* 真實最後一張 clip feature
            real_img_end = real_video[b,:,self.seq_length-1,:,:]
            image = to_pil(real_img_end)
            image = self.preprocess(image).unsqueeze(0).to("cuda")
            real_end_feature = self.clip_model.encode_image(image)

            #* [1,0.95,0.9,0.85, ...]
            similar_weight = np.arange(1, 1 - 0.05 * self.seq_length, -0.05)
            similar_weight[similar_weight<0] = 0
            weight_sum =  np.sum(similar_weight)

            #* clip image feature 的 cosine similarity loss
            for i in range(20):
                begin_sim_loss += similar_weight[i]*F.cosine_similarity(real_begin_feature,image_features_begin[i])
                end_sim_loss += similar_weight[i]*F.cosine_similarity(real_end_feature,image_features_end[i])
                if i<19:
                    between_sim_loss += F.cosine_similarity(image_features_begin[i],image_features_begin[i+1])
                    between_sim_loss += F.cosine_similarity(image_features_end[i],image_features_end[i+1])

            #* normalize 回cosine 範圍 (-1,1)
            begin_sim_loss /= weight_sum
            end_sim_loss /= weight_sum
            between_sim_loss /= (20*2-2)
        
        begin_sim_loss /= batch_num
        end_sim_loss /= batch_num
        between_sim_loss /= batch_num

        #* 1-cosine => 越相近loss 越小
        return 1-begin_sim_loss,1-end_sim_loss,1-between_sim_loss
'''
    