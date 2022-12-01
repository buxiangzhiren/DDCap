import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from tf_adpt_grad import GPT2LMHeadModel
from tqdm import tqdm
import random
import datetime
import pickle
import sys
import argparse
import io, base64, json
from typing import Tuple, Optional, Union
from clip1 import clip
from clip1.clip import _transform
import math
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import skimage.io as io1
from PIL import Image, ImageFilter, ImageOps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from timm.models.layers import trunc_normal_
import numpy as np
from lr_scheduler import build_scheduler
from misc import generate2_adpt_if, evaluate_on_coco_caption
import torch.nn.functional as F


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x):
    return log_x.argmax(1)


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)

    def pad_tokens(self, item: int):
        tokens = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        gt = torch.cat((self.captions_tokens[item], torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            gt = torch.cat((gt, torch.zeros(padding, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            gt = gt[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        return tokens, mask, gt

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens, mask, gt = self.pad_tokens(item)
        img_id = self.image_ids[item]
        # train+restval
        filename = f"{self.data_root}/train2014/COCO_train2014_{int(img_id):012d}.jpg"
        try:
            image = io1.imread(filename)
        except:
            filename = f"{self.data_root}/val2014/COCO_val2014_{int(img_id):012d}.jpg"
            image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        ## this is for pre-computed clip feature
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)
        return tokens, mask, image, gt

    def __init__(self, data_root: str, data_path: str, gpt2_type: str = "gpt2", len_seq=20, normalize_prefix=False):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.normalize_prefix = normalize_prefix
        self.preprocess = _transform(224)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()
        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption['caption'] for caption in captions_raw]
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption2embedding, self.max_seq_len = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(torch.tensor(self.tokenizer.encode(caption['caption']), dtype=torch.int64))
                self.caption2embedding.append(caption["clip_embedding"])  # just index
                max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len], f)
        all_len = torch.tensor([len(self.captions_tokens[i]) for i in range(len(self))]).float()
        ## notice current one is 40
        # self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
        self.max_seq_len = len_seq


class ClipCocoValDataset(Dataset):

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, item: int):
        _filename = self.files[item]
        filename = f"{self.data_root}/val2014/{_filename}"
        for x in self.annotation:
            if 'COCO_val2014_' + str(x['image_id']).zfill(12) + '.jpg' == _filename:
                caption = x['caption']
                break
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        tokens = torch.cat((tokens, torch.tensor(self.tokenizer.encode('<|endoftext|>'))), dim=0)
        gt = torch.clone(tokens)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            gt = torch.cat((gt, torch.zeros(padding, dtype=torch.int64) - 1))  # we set target == -1 as ignore target
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            gt = gt[:self.max_seq_len]
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        image = io1.imread(filename)
        image = Image.fromarray(image)
        image = self.preprocess(image)
        return image, _filename, tokens, gt, mask

    def __init__(self, data_root: str):
        self.data_root = data_root
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.annotation = json.load(open("./MSCOCO_Caption/annotations/captions_val2014.json", "r"))["annotations"]
        with open('captioneval/coco_test.txt') as f:
            self.files = f.read().splitlines()
        self.preprocess = _transform(224)
        self.max_seq_len = 20


class MLP(nn.Module):

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            self._trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            self._trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        for m in self.modules():
            self._init_weights(m)


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec:  # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(
                    TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)


class TransformerMapper(nn.Module):

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, mask_tokens: torch.Tensor, prefix: torch.Tensor,
                mask: Optional[torch.Tensor] = None, t = None,
                labels: Optional[torch.Tensor] = None):
        # tokens = torch.where(mask_tokens == 50257, tokens, mask_tokens) # if you want to use beta, add this line
        embedding_text = self.gpt.transformer.wte(tokens)
        batch_size = embedding_text.size()[0]
        seq_len = embedding_text.size()[1]
        bos_token_embedding = self.bos_embedding.unsqueeze(0).unsqueeze(0).repeat_interleave(repeats=batch_size, dim=0)
        bos_token_embedding = bos_token_embedding.repeat_interleave(repeats=seq_len, dim=1)
        mask_tokens = mask_tokens.unsqueeze(dim=2).repeat_interleave(repeats=self.gpt_embedding_size, dim=2)
        embedding_text = torch.where(mask_tokens == 50257, bos_token_embedding, embedding_text)
        prefix, len_cls = self.image_encode(prefix)
        prefix_projections = self.clip_project(prefix)
        if self.training:
            empty_idx = int(self.if_drop_rate * batch_size)
            for i_image in range(empty_idx):
                prefix_projections[i_image,:,:] = self.pad_embedding
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(t=t, inputs_embeds=embedding_text, labels=labels, attention_mask=mask,
                       encoder_hidden_states=prefix_projections)
        return out, self.len_head(torch.detach(len_cls))

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP, Timestep: int = 20, if_drop_rate=0.02):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        # self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        configuration = GPT2Config.from_pretrained('gpt2')
        configuration = configuration.__dict__.copy()
        configuration.update({'scale_attn_by_inverse_layer_idx': False})
        configuration.update({'reorder_and_upcast_attn': False})
        configuration.update({'use_cache': False})
        configuration = GPT2Config(**configuration)
        self.gpt = GPT2LMHeadModel(configuration)
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
        self.clip_model.requires_grad_(False)
        self.clip_model.visual.requires_grad_(True)
        self.clip_model.visual.ln_post.requires_grad_(False)
        self.clip_model.visual.proj.requires_grad_(False)
        self.image_encode = self.clip_model.encode_image
        self.time_step = Timestep
        self.if_drop_rate = if_drop_rate
        self.num_classes = configuration.vocab_size + 1
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.bos_embedding = nn.Parameter(torch.randn(self.gpt_embedding_size)) # used as the vector of mask token
        self.pad_embedding = nn.Parameter(torch.randn(size=(196, 768), requires_grad=True, dtype=torch.float64)) # image free vector
        self.len_head = MLP((512, self.gpt_embedding_size // 2, 20))
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((768, self.gpt_embedding_size // 2,
                                     self.gpt_embedding_size))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)
        at, bt, ct, att, btt, ctt = alpha_schedule(self.time_step, N=self.num_classes)  # alpha schedule
        at = torch.tensor(at.astype('float64'))
        bt = torch.tensor(bt.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype('float64'))
        btt = torch.tensor(btt.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5

        self.diffusion_acc_list = [0] * self.time_step
        self.diffusion_keep_list = [0] * self.time_step
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt', log_bt.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt', log_cumprod_bt.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())

    def q_pred(self, log_x_start, t):  # q(xt|x0)
        # log_x_start can be onehot or not
        t = (t + (self.time_step + 1)) % (self.time_step + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(self.log_cumprod_bt, t, log_x_start.shape)  # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        # log_probs = torch.zeros(log_x_start.size()).type_as(log_x_start)
        p1 = log_add_exp(log_x_start[:, :-1, :] + log_cumprod_at, log_cumprod_bt)
        p2 = log_add_exp(log_x_start[:, -1:, :] + log_1_min_cumprod_ct, log_cumprod_ct)

        return torch.cat([p1, p2], dim=1)

    def log_sample_categorical(self, logits):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)

        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def q_posterior(self, log_x_start, log_x_t, t):  # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0)*p(x0|xt))
        # notice that log_x_t is onehot
        # log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
        #                       = log(p(x0|xt)) + log(q(xt|xt_1,x0)) + log(q(xt_1|x0)) - log(q(xt|x0))  (*)
        assert t.min().item() >= 0 and t.max().item() < self.time_step
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes - 1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector + 1.0e-30).expand(-1, -1, log_x_start.size(-1))

        # log(q(xt|x0))
        log_qt = self.q_pred(log_x_t, t)
        log_qt = torch.cat((log_qt[:, :-1, :], log_zero_vector), dim=1)
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, self.num_classes - 1, -1)
        ct_cumprod_vector = torch.cat((ct_cumprod_vector, log_one_vector), dim=1)
        log_qt = (~mask) * log_qt + mask * ct_cumprod_vector

        # log(q(xt|xt_1,x0))
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:, :-1, :], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, self.num_classes - 1, -1)
        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask) * log_qt_one_timestep + mask * ct_vector

        q = log_x_start - log_qt  # log(p(x0|xt)/q(xt|x0))
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp  # norm(log(p(x0|xt)/q(xt|x0)))  to leverage self.q_pred
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q,
                                                         t - 1) + log_qt_one_timestep + q_log_sum_exp  # get (*), last term is re-norm
        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def q_pred_one_timestep(self, log_x_t, t):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(self.log_bt, t, log_x_t.shape)  # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        # log_probs = torch.zeros(log_x_t.size()).type_as(log_x_t)
        p1 = log_add_exp(log_x_t[:, :-1, :] + log_at, log_bt)
        p2 = log_add_exp(log_x_t[:, -1:, :] + log_1_min_ct, log_ct)

        return torch.cat([p1, p2], dim=1)


def alpha_schedule(time_step, N=100, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step - 1) * (att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:] / att[:-1]
    ctt = np.arange(0, time_step) / (time_step - 1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1 - one_minus_ct
    bt = (1 - at - ct) / N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1 - att - ctt) / N
    return at, bt, ct, att, btt, ctt


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.tag}-args.json")
    if args.local_rank == 0:
        with open(out_path, 'w') as outfile:
            json.dump(config, outfile)


def load_model(model, args, epoch_or_latest: Union[str, int] = '_latest'):
    # with open(config_path) as f:
    #     config = json.load(f)
    # parser = argparse.ArgumentParser()
    # parser.set_defaults(**config)
    # args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.tag}{epoch_or_latest}.pt")
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["model"])
    else:
        print(f"{model_path} is not exist")
    return model


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, clip_lr, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []
    has_decay1 = []
    no_decay1 = []
    has_decay_name1 = []
    no_decay_name1 = []
    for name, param in model.clip_model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay1.append(param)
            no_decay_name1.append('clip_model.' + name)
        else:
            has_decay1.append(param)
            has_decay_name1.append('clip_model.' + name)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            if name not in no_decay_name1:
                no_decay.append(param)
                no_decay_name.append(name)
        else:
            if name not in has_decay_name1:
                has_decay.append(param)
                has_decay_name.append(name)
    print(f'No decay params: {no_decay_name}')
    print(f'Has decay params: {has_decay_name}')
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.},
            {'params': has_decay1, 'lr': clip_lr},
            {'params': no_decay1, 'weight_decay': 0., 'lr': clip_lr},
            ]


@torch.no_grad()
def val(model, epoch, val_dataloader, args):
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f">>> Evaling epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(val_dataloader), desc=args.tag)
    result_all = []
    val_loss_all = {}
    for i_vl in range(model.module.time_step):
        val_loss_all[f'vl_loss_{i_vl}'] = []
    for idx, (image, image_path, tokens, gt, mask) in enumerate(val_dataloader):
        image = image.cuda(non_blocking=True)
        tokens = tokens.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        for i_t in range(0, model.module.time_step):
            b, device = mask.size()[0], mask.device
            t = torch.tensor(i_t, device=device).long().repeat_interleave(repeats=b)
            log_x_start = index_to_log_onehot(tokens, model.module.num_classes)
            log_xt = model.module.q_sample(log_x_start=log_x_start, t=t)
            xt = log_onehot_to_index(log_xt)
            mask_tokens = xt
            ex_mask = torch.zeros_like(mask_tokens) - 10000
            ex_nomask = torch.zeros_like(mask_tokens)
            all_mask = torch.where(mask_tokens == 50257, ex_mask, ex_nomask)
            all_mask = all_mask.unsqueeze(dim=1).repeat_interleave(repeats=mask_tokens.size(1), dim=1)
            for each_b in range(all_mask.size(0)):
                for each_token in range(all_mask[each_b].size(0)):
                    all_mask[each_b, each_token, each_token] = 0
            padding_mask = mask.unsqueeze(dim=1)
            padding_mask = (1.0 - padding_mask.long()) * -10000
            all_mask = torch.clamp(all_mask + padding_mask, -10000, 0)
            outputs, len_out = model(tokens, mask_tokens, image, all_mask, t)
            logits = outputs.logits
            loss_len = nnf.cross_entropy(len_out, mask.sum(dim=-1).to(torch.long) - 1)
            loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten(), ignore_index=-1)
            val_loss_all[f'vl_loss_{i_t}'].append(loss)
        prefix, len_cls = model.module.image_encode(image)
        prefix_embed = model.module.clip_project(prefix)
        len_pre = model.module.len_head(len_cls)
        if args.use_beam_search:
            assert False, "Not check beam search for now"
            generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
        else:
            generated_text_prefix = generate2_adpt_if(model, tokenizer, embed=prefix_embed, len_pre=len_pre.argmax(-1) + 1)

        torch.cuda.synchronize()
        progress.update()

        r = [{'image_id': _image_path, 'result': _text} for _image_path, _text in
             zip(image_path, generated_text_prefix)]
        result_all.extend(r)
    progress.close()
    os.makedirs(f'.cache/{args.tag}', exist_ok=True)
    json.dump(result_all, open(f".cache/{args.tag}/tmp-results-{dist.get_rank()}.json", "w"))
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        result_all = []
        ra_id = []
        for i in range(dist.get_world_size()):
            part_result = json.load(open(f".cache/{args.tag}/tmp-results-{i}.json"))
            for ep in part_result:
                if ep['image_id'] not in ra_id:
                    ra_id.append(ep['image_id'])
                    result_all.append(ep)
        result = evaluate_on_coco_caption(result_all,
                                          os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}-results.json"),
                                          os.path.join(args.data_root, 'annotations/captions_val2014.json'))
    else:
        result = None
    torch.distributed.barrier()
    if dist.get_rank() == 0:
        log_dict = {}
        for key in val_loss_all.keys():
            log_dict[key] = torch.tensor(val_loss_all[key]).mean().item()
        print(log_dict)
    return result


def sample_time(b, num_timesteps, device):
    t = torch.randint(0, num_timesteps, (b,), device=device).long()
    pt = torch.ones_like(t).float() / num_timesteps
    return t, pt


def multinomial_kl(log_prob1, log_prob2):  # compute KL loss on log_prob
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args,
          output_dir: str = ".", output_prefix: str = ""):
    model.train()
    num_steps = len(train_dataloader)

    train_dataloader.sampler.set_epoch(epoch)
    print(f">>> Training epoch {epoch}")
    sys.stdout.flush()
    progress = tqdm(total=len(train_dataloader), desc=output_prefix)
    for idx, (tokens, mask, prefix, gt) in enumerate(train_dataloader):
        # prefix is raw images
        tokens = tokens.cuda(non_blocking=True)
        mask = mask.cuda(non_blocking=True)
        prefix = prefix.cuda(non_blocking=True)
        gt = gt.cuda(non_blocking=True)
        b, device = mask.size()[0], mask.device
        t, pt = sample_time(b, torch.tensor(args.time_step).to(torch.int), device)
        # add noise
        log_x_start = index_to_log_onehot(tokens, model.module.num_classes)
        log_xt = model.module.q_sample(log_x_start=log_x_start, t=t)
        xt = log_onehot_to_index(log_xt)
        mask_tokens = xt

        # generate concentrate mask attention
        ex_mask = torch.zeros_like(mask_tokens) - 10000
        ex_nomask = torch.zeros_like(mask_tokens)
        all_mask = torch.where(mask_tokens == 50257, ex_mask, ex_nomask)
        all_mask = all_mask.unsqueeze(dim=1).repeat_interleave(repeats=mask_tokens.size(1), dim=1)
        for each_b in range(all_mask.size(0)):
            for each_token in range(all_mask[each_b].size(0)):
                all_mask[each_b, each_token, each_token] = 0
        padding_mask = mask.unsqueeze(dim=1)
        padding_mask = (1.0 - padding_mask.long()) * -10000
        all_mask = torch.clamp(all_mask + padding_mask, -10000, 0)

        # predict x0
        with amp.autocast(enabled=args.enable_amp):
            outputs, len_out = model(tokens, mask_tokens, prefix, all_mask, t)
        logits = outputs.logits
        loss_len = nnf.cross_entropy(len_out, mask.sum(dim=-1).to(torch.long) - 1)
        loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt.flatten(), ignore_index=-1)
        loss = loss + loss_len
        optimizer.zero_grad()
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)
        torch.cuda.synchronize()
        progress.set_postfix({"loss": loss.item() - loss_len.item(), 'lr': optimizer.param_groups[0]['lr'], 'loss_len':loss_len.item(), "loss_all": loss.item()})
        progress.update()
    progress.close()
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data/coco/oscar_split_ViT-B_32_train_512.pkl')
    parser.add_argument('--data_root', default='./MSCOCO_Caption/', help='raw coco training image path')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--clip_lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=0.01)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', default='debug',
                        help='tag of job, used for wandb and output')
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--use_beam_search', action='store_true')
    parser.add_argument('--enable-amp', action='store_true')
    parser.add_argument('--time_step',  type=int, default=20)
    parser.add_argument('--loss_weight', type=float, default=[0.5, 0.5])
    parser.add_argument('--if_drop_rate', type=float, default=0.1)
    parser.add_argument('--disable-amp', action='store_false', dest='enable_amp')
    parser.set_defaults(enable_amp=True)

    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir, args.tag)
    os.makedirs(args.out_dir, exist_ok=True)
    save_config(args)
    return args


def main(args):
    prefix_dim = 640 if args.is_rn else 512
    args.mapping_type = {'mlp': MappingType.MLP, 'transformer': MappingType.Transformer}[args.mapping_type]
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=args.mapping_type, Timestep=args.time_step,
                                  if_drop_rate=args.if_drop_rate)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(args.prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                 num_layers=args.num_layers, mapping_type=args.mapping_type, Timestep=args.time_step,
                                 if_drop_rate=args.if_drop_rate)
        print("Train both prefix and GPT")
        sys.stdout.flush()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_params:,} total parameters')

    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    # ckpt = torch.load('caption_diff_pretrain_CC-018.pt', map_location="cpu")
    # model.load_state_dict(ckpt["model"])
    model = model.cuda()

    parameters = get_pretrain_param_groups(model, args.lr * args.clip_lr)
    optimizer = AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    scaler = amp.GradScaler()
    model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    dataset = ClipCocoDataset(args.data_root, args.data, normalize_prefix=args.normalize_prefix)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    train_dataloader = DataLoader(dataset, batch_size=args.bs, sampler=train_sampler, num_workers=8, pin_memory=True, drop_last=True)

    val_dataset = ClipCocoValDataset(args.data_root)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, sampler=val_sampler, num_workers=8, pin_memory=True, drop_last=False)

    lr_args = {"LR_SCHEDULER_NAME": "cosine", "EPOCHS": args.epochs, "WARMUP_EPOCHS": 5, "MIN_LR": 1e-6,
               "WARMUP_LR": 1e-7}
    lr_scheduler = build_scheduler(lr_args, optimizer, len(train_dataloader))

    best_cider = 0
    for epoch in range(args.epochs):
        _ = train(model, epoch, train_dataloader, optimizer, lr_scheduler, scaler, args, output_dir=args.out_dir, output_prefix=args.tag)
        result = val(model, epoch, val_dataloader, args)
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            if dist.get_rank() == 0:
                torch.save(
                    {'model':model.module.state_dict(), 'lr_scheduler': lr_scheduler.state_dict(), 'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict()},
                    os.path.join(args.out_dir, f"{args.tag}-{epoch:03d}.pt"),
                )
        if dist.get_rank() == 0 and result['CIDEr'] > best_cider:
            best_cider = result['CIDEr']
            torch.save(
                {'model':model.module.state_dict()},
                os.path.join(args.out_dir, f"{args.tag}-best.pt"),
            )

if __name__ == '__main__':
    # command:  python -m torch.distributed.launch --nproc_per_node 4 train.py --data ./oscar_split_ViT-B_32_train_512.pkl --out_dir ./output --bs 32
    args = parse_args()
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    dist.init_process_group("nccl", init_method='env://', rank=args.local_rank, world_size=world_size)
    torch.distributed.barrier()
    setup_for_distributed(args.local_rank == 0)  ##### HERE

    seed = dist.get_rank() + args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    main(args)
