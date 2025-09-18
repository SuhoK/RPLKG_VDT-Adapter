import os.path as osp
import os

import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from . import clip_custom as clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


CUSTOM_TEMPLATES = {
    'OxfordPets': 'a photo of a {}, a type of pet.',
    'OxfordFlowers': 'a photo of a {}, a type of flower.',
    'FGVCAircraft': 'a photo of a {}, a type of aircraft.',
    'DescribableTextures': '{} texture.',
    'EuroSAT': 'a centered satellite photo of {}.',
    'StanfordCars': 'a photo of a {}.',
    'Food101': 'a photo of {}, a type of food.',
    'SUN397': 'a photo of a {}.',
    'Caltech101': 'a photo of a {}.',
    'UCF101': 'a photo of a person doing {}.',
    'ImageNet': 'a photo of a {}.',
    'ImageNetSketch': 'a photo of a {}.',
    'ImageNetV2': 'a photo of a {}.',
    'ImageNetA': 'a photo of a {}.',
    'ImageNetR': 'a photo of a {}.',
    'CUB': 'a photo of {}, a type of bird.',
}

gpt4_filename = {
    "OxfordPets": 'oxford_pets.pt',
    "CUB": 'cub.pt',
    "OxfordFlowers": 'oxford_flowers.pt',
    "FGVCAircraft": 'fgvc_aircraft.pt',
    "DescribableTextures":  'dtd.pt',
    "EuroSAT":  "eurosat.pt",
    "StanfordCars": "stanford_cars.pt",
    "Food101": "food-101.pt",
    "SUN397": "sun397.pt",
    "Caltech101": "caltech-101.pt",
    "UCF101": "ucf101.pt",
    "ImageNet": "imagenet.pt",
}

##### resize ######
def resize_pos_embed(pos_embed, new_grid_size):
    """
    Resize positional embeddings by interpolating the grid part.
    """
    if pos_embed.ndim == 2:
        pos_embed = pos_embed.unsqueeze(0)

    cls_pos = pos_embed[:, :1, :]
    grid_pos = pos_embed[:, 1:, :]  # shape: (1, N, D)

    num_patches = grid_pos.shape[1]
    old_grid_size = int(num_patches ** 0.5)
    assert old_grid_size * old_grid_size == num_patches, "Grid is not square."

    grid_pos = grid_pos.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
    grid_pos = F.interpolate(grid_pos, size=(new_grid_size, new_grid_size), mode='bicubic', align_corners=False)
    grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(1, new_grid_size * new_grid_size, -1)

    return torch.cat([cls_pos, grid_pos], dim=1)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())
    #####  positional embedding resize #####
    if 'vit' in backbone_name.lower():
        try:
            patch_size = model.visual.patch_size
        except AttributeError:
            try:
                patch_size = model.visual.patch_embedding.patch_size
            except AttributeError:
                patch_size = model.visual.conv1.kernel_size[0]  # fallback
                
        new_size = cfg.INPUT.SIZE[0]
        new_grid_size = new_size // patch_size

        pos_embed = model.visual.positional_embedding
        if pos_embed.ndim == 2:
            pos_embed = pos_embed.unsqueeze(0)

        pos_embed = resize_pos_embed(pos_embed, new_grid_size)
        model.visual.positional_embedding = nn.Parameter(pos_embed)
    # ================================ #
    return model


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, 
                 dropout=0.1, ratio=0.5):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        nn.init.xavier_normal_(self.w_qs.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_ks.weight, gain=1.0)
        nn.init.xavier_normal_(self.w_vs.weight, gain=0.67)


        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model) # feed forward layer
        nn.init.xavier_normal_(self.fc.weight, gain=0.67)

        self.dropout = nn.Dropout(dropout)

        self.ratio = ratio

        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(2*(self.ratio * output + (1-self.ratio) * residual))

        return output


class SelfAttnAdapter(nn.Module):

    def __init__(self, c_in, reduction=4, ratio=0.5):
        super(SelfAttnAdapter, self).__init__()
        self.attn = MultiHeadAttention(1, c_in, 
            c_in//reduction, c_in//reduction, dropout=0.5, ratio=ratio) #.cuda()

    def forward(self, x):
        device = x.device
        self.attn.to(device)
        x = self.attn(x, x, x)
        return x




class TextEncoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to('cuda')
        text_features = self.clip_model.encode_text(prompts)
        x = text_features
        return x




class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames
        model_dim = model_dim = clip_model.text_projection.shape[1] #clip_model.visual.transformer.width

        we_adapter = cfg.TRAINER.CLIP_ADAPTER.WORD_ADAPTER_TYPE
        self.ratio = cfg.TRAINER.CLIP_ADAPTER.RATIO

        if cfg.MODEL.BACKBONE.NAME == 'RN50':
            model_dim = 1024
            
        elif cfg.MODEL.BACKBONE.NAME.lower() == 'vit-l/14':
            self.input_dim = 768
            
        else:
            model_dim = 512


        if 'linear' in we_adapter:
            self.adapter = Adapter(model_dim, 4)
        elif we_adapter == 'self_attn':
            ratio = self.ratio
            self.adapter = SelfAttnAdapter(model_dim, 4, ratio=ratio)


        self.attr = None
        
        if we_adapter is not None:
            print(f'Using {we_adapter} adapter')
            gpt4_sentences = torch.load(f'./gpt4_data/{gpt4_filename[cfg.DATASET.NAME]}')
            print('gpt4 sentences ', gpt4_sentences)
            
            attr = []
            for cl in classnames:
                # (OxfordFlowers·StanfordCars·EuroSAT 는 공백 유지,
                #  나머지는 기존 로직 대로 '_' 로 join)
                if cfg.DATASET.NAME not in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
                    cl_for_lookup = '_'.join(cl.split(' '))
                else:
                    cl_for_lookup = cl
            
                key = cl_for_lookup.lower()          # ← 바로 여기!
            
                if key not in gpt4_sentences:        # 안전 확인
                    raise ValueError(f"[GPT prompt 누락] '{key}' 가 gpt4_sentences에 없습니다.")
            
                current_sentences = gpt4_sentences[key]
            
                # 토크나이즈 및 feature 추출
                current_sentences = torch.cat([clip.tokenize(c) for c in current_sentences])
                current_sentences = current_sentences.to('cuda')
                clip_model = clip_model.to('cuda')
                with torch.no_grad():
                    current_text_features = clip_model.encode_text(current_sentences)
            
                attr.append(current_text_features)
            # now get the text features for all the gpt4 sentences
#            for cl in classnames:
                # need to include code for all datasets, some dont need the folowing line
#                if cfg.DATASET.NAME in ['OxfordFlowers', 'StanfordCars', 'EuroSAT']:
#                    pass
#                else:
#                    cl = '_'.join(cl.split(' '))
#                current_sentences = gpt4_sentences[cl.lower()]
#                current_sentences = torch.cat([clip.tokenize(c) for c in current_sentences])
#                current_sentences = current_sentences.to('cuda')
#                clip_model = clip_model.to('cuda')
#                with torch.no_grad():
#                    current_text_features = clip_model.encode_text(current_sentences)
#                    attr.append(current_text_features)
            attr = torch.stack(attr)
            #self.register_buffer('attr', attr.cpu()) #self.attr = attr
            if hasattr(self, 'attr'):
                delattr(self, 'attr')
            self.register_buffer('attr', attr.cpu())

        self.we_adapter = we_adapter


            
    def forward(self, image):

        device = image.device
        
        image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.attr.to(device)

        #text_features = self.attr

        if self.we_adapter == 'linear':
            text_features = text_features.mean(dim=1)

            text_features = self.adapter(text_features)


        elif self.we_adapter == 'linear_residual':
            text_features = text_features.mean(dim=1)

            x = self.adapter(text_features)

            ratio = self.ratio
            text_features = ratio * x + (1 - ratio) * text_features
        
        elif self.we_adapter == 'self_attn':
            text_features = self.adapter(text_features)
            text_features = text_features.mean(dim=1)



        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits


@TRAINER_REGISTRY.register()
class CLIP_Adapter_gpt(TrainerX):
    """ CLIP-Adapter with gpt generated prompts"""

    def build_model(self):
        cfg = self.cfg

        # 1) classnames 가져오기 및 정규화
        classnames = getattr(self.dm.dataset, "classnames", None)
        if isinstance(classnames, dict):
            classnames = [classnames[i] for i in sorted(classnames)]

        norm_names = []
        for c in classnames:
            if isinstance(c, str) and c.isdigit():
                c = int(c)
            if isinstance(c, int):
                if hasattr(self.dm.dataset, "classnames") and c < len(self.dm.dataset.classnames):
                    norm_names.append(self.dm.dataset.classnames[c])
                elif hasattr(self.dm.dataset, "lab2cname") and c in getattr(self.dm.dataset, "lab2cname", {}):
                    norm_names.append(self.dm.dataset.lab2cname[c])
                else:
                    norm_names.append(str(c))
            else:
                norm_names.append(str(c))
        classnames = [str(c) for c in norm_names]

        # 2) gpt4_sentences 로드 + 키 소문자화 + 누락 보충
        global gpt4_sentences
        gpt4_sentences = torch.load(f'./gpt4_data/{gpt4_filename[cfg.DATASET.NAME]}')
        _g = {str(k).lower(): v for k, v in gpt4_sentences.items()}

        missing = []
        for cname in classnames:
            key = cname.lower()
            if key not in _g:
                missing.append(key)
                _g[key] = f"a photo of a {cname}"
        if missing:
            print(f"[WARN] GPT prompts missing for: {missing}. Filled with default template.")

        for idx, cname in enumerate(classnames):
            key = cname.lower()
            prompt = _g.get(key, f"a photo of a {cname}")
            _g[str(idx)] = prompt
            _g[idx] = prompt

        gpt4_sentences = _g  # 전역에 반영

        print('class names length ', len(classnames))

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.float()

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print('Turning off gradients in both the image and the text encoder')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model('clip_adapter_gpt', self.model.adapter, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)


    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']
            
            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']
            
            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
