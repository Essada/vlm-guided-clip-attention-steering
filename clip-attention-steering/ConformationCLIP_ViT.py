import torch
import torch.nn as nn
import open_clip
from torch.nn import functional as F


class CLIPViT(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.preprocess = preprocess
        self.tokenizer  = open_clip.get_tokenizer(model_name)

        for p in model.parameters():
            p.requires_grad = False

        self.visual          = model.visual
        self._encode_text_fn = model.encode_text

        self.model_name = model_name
        self.num_layers = len(self.visual.transformer.resblocks)
        self.num_heads  = self.visual.transformer.resblocks[0].attn.num_heads
        self.embed_dim  = self.visual.transformer.resblocks[0].attn.embed_dim

        pos_tokens = self.visual.positional_embedding.shape[0] - 1
        self.grid_size = int(round(pos_tokens ** 0.5))
        self.num_spatial_tokens = self.grid_size * self.grid_size
        patch = self.visual.conv1.kernel_size
        self.patch_size = patch[0] if isinstance(patch, tuple) else patch

    @torch.no_grad()
    def encode_text(self, texts):
        tokens = self.tokenizer(texts).to(next(self.parameters()).device)
        return F.normalize(self._encode_text_fn(tokens), dim=-1)

    def _prepare_tokens(self, images):
        v   = self.visual
        x   = v.conv1(images)
        x   = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + v.positional_embedding.to(x.dtype)
        x   = v.ln_pre(x)
        return x

    def _pool_and_project(self, x):
        v = self.visual
        x = v.ln_post(x[:, 0])
        if v.proj is not None:
            x = x @ v.proj
        return x

    def _run_transformer_w_attention(self, x):
        attns = []
        for block in self.visual.transformer.resblocks:
            x_norm = block.ln_1(x)
            attn_out, attn_w = block.attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False,
            )
            attns.append(attn_w.detach())
            x = x + block.ls_1(attn_out)
            x = x + block.ls_2(block.mlp(block.ln_2(x)))
        return x, attns

    def _run_transformer_pasta(self, x, attns, token_idx_per_image, alpha,
                               target_layer_heads, return_attns=False):
        heads_per_layer = {}
        for l, h in target_layer_heads:
            heads_per_layer.setdefault(l, []).append(h)

        steered_attns = [] if return_attns else None

        for layer_idx, block in enumerate(self.visual.transformer.resblocks):
            x_norm = block.ln_1(x)

            if layer_idx not in heads_per_layer:
                if return_attns:
                    attn_out, layer_attn = block.attn(
                        x_norm, x_norm, x_norm,
                        need_weights=True,
                        average_attn_weights=False,
                    )
                    steered_attns.append(layer_attn.detach())
                else:
                    attn_out, _ = block.attn(x_norm, x_norm, x_norm, need_weights=False)
            else:
                if return_attns:
                    attn_out, layer_steered_attn = self._steered_block_output(
                        block, x_norm, token_idx_per_image,
                        alpha, heads_per_layer[layer_idx],
                        return_attn=True,
                    )
                    steered_attns.append(layer_steered_attn)
                else:
                    attn_out = self._steered_block_output(
                        block, x_norm, token_idx_per_image,
                        alpha, heads_per_layer[layer_idx],
                    )

            x = x + block.ls_1(attn_out)
            x = x + block.ls_2(block.mlp(block.ln_2(x)))

        if return_attns:
            return x, steered_attns
        return x

    def _steered_block_output(self, block, x_norm, token_idx_per_image,
                               alpha, steer_heads, return_attn=False):
        N, seq_len, embed_dim = x_norm.shape
        num_heads = block.attn.num_heads
        head_dim  = embed_dim // num_heads

        in_w = block.attn.in_proj_weight
        in_b = block.attn.in_proj_bias

        q = F.linear(
            x_norm,
            in_w[:embed_dim],
            in_b[:embed_dim] if in_b is not None else None,
        )
        k = F.linear(
            x_norm,
            in_w[embed_dim:2 * embed_dim],
            in_b[embed_dim:2 * embed_dim] if in_b is not None else None,
        )
        v = F.linear(
            x_norm,
            in_w[2 * embed_dim:],
            in_b[2 * embed_dim:] if in_b is not None else None,
        )

        q = q.reshape(N, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(N, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(N, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
        steered = attn.softmax(dim=-1)

        steered = steered.clone()
        for bi in range(N):
            selected_spatial = set(token_idx_per_image[bi])
            if not selected_spatial or len(selected_spatial) >= self.num_spatial_tokens:
                continue

            all_keys = set(range(seq_len))
            g = [0] + [t + 1 for t in selected_spatial]
            g_minus = list(all_keys - set(g))
            CLS_ROW_ONLY = True
            if CLS_ROW_ONLY:
                for h in steer_heads:
                    A_cls = steered[bi, h, 0].clone()
                    Ci = A_cls[g].sum() + alpha * A_cls[g_minus].sum()
                    Ci = Ci.clamp(min=1e-6)
                    steered[bi, h, 0, g]       = A_cls[g]               / Ci
                    steered[bi, h, 0, g_minus] = alpha * A_cls[g_minus] / Ci
            else:
                for h in steer_heads:
                    A  = steered[bi, h].clone()
                    Ci = A[:, g].sum(dim=-1) + alpha * A[:, g_minus].sum(dim=-1)
                    Ci = Ci.clamp(min=1e-6)
                    steered[bi, h, :, g]       = A[:, g]               / Ci.unsqueeze(-1)
                    steered[bi, h, :, g_minus] = alpha * A[:, g_minus] / Ci.unsqueeze(-1)

        H = steered @ v
        H = H.permute(0, 2, 1, 3).reshape(N, seq_len, embed_dim)
        H = F.linear(H, block.attn.out_proj.weight, block.attn.out_proj.bias)
        if return_attn:
            return H, steered
        return H

    @torch.no_grad()
    def encode_image(self, images):
        tokens    = self._prepare_tokens(images)
        out, _    = self._run_transformer_w_attention(tokens)
        return F.normalize(self._pool_and_project(out), dim=-1)

    @torch.no_grad()
    def encode_image_pasta(self, images, token_idx_per_image, alpha,
                           target_layer_heads):
        tokens   = self._prepare_tokens(images)
        out      = self._run_transformer_pasta(tokens, None, token_idx_per_image,
                                               alpha, target_layer_heads)
        return F.normalize(self._pool_and_project(out), dim=-1)

    def forward(self, image=None, text=None):
        image_features = self.encode_image(image) if image is not None else None
        text_features  = self.encode_text(text)   if text  is not None else None
        return image_features, text_features
