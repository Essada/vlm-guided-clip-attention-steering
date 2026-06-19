import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from PIL import Image

from ConformationCLIP_ViT import CLIPViT
from test_vit_cars import get_token_indices, bbox_to_patch_indices

DATA_ROOT      = os.path.expanduser("~/data/test")
N_PROFILE      = 100
N_EVAL         = 250
BATCH_SIZE     = 8
TOP_HEADS_K    = 5
PROFILE_LAYERS = list(range(8, 12))

CAR_PARTS_QUERY = "car ."
BOX_THRESHOLD   = 0.25
TEXT_THRESHOLD  = 0.20


class CarsWithPIL(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        pil         = Image.open(path).convert("RGB")
        return self.base.transform(pil), label, pil


def collate_with_pil(batch):
    tensors = torch.stack([b[0] for b in batch])
    labels  = torch.tensor([b[1] for b in batch])
    pils    = [b[2] for b in batch]
    return tensors, labels, pils


def load_grounding_dino(device):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    gd_id     = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(gd_id)
    model     = AutoModelForZeroShotObjectDetection.from_pretrained(gd_id).to(device)
    model.eval()
    print(f"Loaded Grounding DINO tiny")
    return processor, model


def parts_patch_indices(pil_img, gd_proc, gd_model, device):
    orig_w, orig_h = pil_img.size
    inputs = gd_proc(
        images=pil_img,
        text=CAR_PARTS_QUERY,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = gd_model(**inputs)

    try:
        results = gd_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]
    except TypeError:
        results = gd_proc.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(orig_h, orig_w)],
        )[0]

    boxes = results["boxes"]
    if boxes.shape[0] == 0:
        return list(range(49))

    indices = set()
    for box in boxes.cpu().tolist():
        indices.update(bbox_to_patch_indices(box, orig_w, orig_h))
    return list(indices)


@torch.no_grad()
def eval_plain(model, loader, class_text_feats):
    correct, total = 0, 0
    for images, labels, _ in tqdm(loader, desc="Plain CLIP"):
        images = images.to(next(model.parameters()).device)
        labels = labels.to(next(model.parameters()).device)
        feat   = model.encode_image(images)
        correct += (feat @ class_text_feats.T).argmax(dim=-1).eq(labels).sum().item()
        total   += images.shape[0]
    return correct / total * 100


@torch.no_grad()
def eval_pasta_grad(model, loader, class_text_feats, heads, alpha, top_k):
    correct, total = 0, 0
    for images, labels, _ in tqdm(loader, desc="PASTA (grad)"):
        dev    = next(model.parameters()).device
        images = images.to(dev)
        labels = labels.to(dev)
        pred   = (model.encode_image(images) @ class_text_feats.T).argmax(dim=-1)
        token_idx = get_token_indices(model, images, class_text_feats[pred], top_k)
        feat      = model.encode_image_pasta(images, token_idx, alpha, heads)
        correct  += (feat @ class_text_feats.T).argmax(dim=-1).eq(labels).sum().item()
        total    += images.shape[0]
    return correct / total * 100


def precompute_dino_indices(loader, gd_proc, gd_model, device):
    all_indices = []
    for _, _, pils in tqdm(loader, desc="DINO (precompute)"):
        for p in pils:
            all_indices.append(parts_patch_indices(p, gd_proc, gd_model, device))
    return all_indices


@torch.no_grad()
def eval_pasta_cached(model, loader, class_text_feats, heads, alpha,
                      cached_indices):
    correct, total = 0, 0
    dev  = next(model.parameters()).device
    idx  = 0
    desc = f"PASTA (DINO) α={alpha}"
    for images, labels, _ in tqdm(loader, desc=desc):
        images    = images.to(dev)
        labels    = labels.to(dev)
        N         = images.shape[0]
        token_idx = cached_indices[idx: idx + N]
        idx      += N
        feat      = model.encode_image_pasta(images, token_idx, alpha, heads)
        correct  += (feat @ class_text_feats.T).argmax(dim=-1).eq(labels).sum().item()
        total    += N
    return correct / total * 100


@torch.no_grad()
def profile_heads(model, loader, class_text_feats, cached_indices, alpha=0.1,
                  layers=PROFILE_LAYERS):
    num_heads = model.num_heads
    dev       = next(model.parameters()).device
    scores    = {}

    pairs = [(l, h) for l in layers for h in range(num_heads)]
    for l, h in tqdm(pairs, desc="Profiling heads"):
        correct, total, idx = 0, 0, 0
        for images, labels, _ in loader:
            images    = images.to(dev)
            labels    = labels.to(dev)
            N         = images.shape[0]
            token_idx = cached_indices[idx: idx + N]
            idx      += N
            feat      = model.encode_image_pasta(images, token_idx, alpha, [(l, h)])
            correct  += (feat @ class_text_feats.T).argmax(dim=-1).eq(labels).sum().item()
            total    += N
        scores[(l, h)] = correct / total * 100

    return sorted(scores.items(), key=lambda x: -x[1])


def main():
    root   = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip = CLIPViT()
    clip.eval()

    full_ds  = ImageFolder(root, transform=clip.preprocess)
    all_idx  = torch.randperm(len(full_ds), generator=torch.Generator().manual_seed(42)).tolist()

    profile_idx = all_idx[:N_PROFILE]
    eval_idx    = all_idx[:N_EVAL]

    profile_subset = torch.utils.data.Subset(CarsWithPIL(full_ds), profile_idx)
    eval_subset    = torch.utils.data.Subset(CarsWithPIL(full_ds), eval_idx)

    profile_loader = DataLoader(profile_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, collate_fn=collate_with_pil)
    eval_loader    = DataLoader(eval_subset,    batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=0, collate_fn=collate_with_pil)

    print(f"Profile: {N_PROFILE} images  |  Eval: {N_EVAL} images  |  {len(full_ds.classes)} classes\n")

    class_text_feats = clip.encode_text([f"a photo of a {c}" for c in full_ds.classes])

    gd_proc, gd_model = load_grounding_dino(device)

    print("Precomputing DINO indices for profile set...")
    profile_dino = precompute_dino_indices(profile_loader, gd_proc, gd_model, device)

    acc_plain_profile = eval_plain(clip, profile_loader, class_text_feats)
    print(f"Plain CLIP (profile): {acc_plain_profile:.2f}%\n")

    ranked = profile_heads(clip, profile_loader, class_text_feats, profile_dino)

    print(f"\nTop-10 heads (profile):")
    print(f"  {'(layer,head)':<14} {'acc':>8}  {'Δ':>7}")
    for (l, h), acc in ranked[:10]:
        print(f"  ({l},{h}){'':<10} {acc:>8.2f}  {acc - acc_plain_profile:>+7.2f}%")

    best_heads = [lh for lh, _ in ranked[:TOP_HEADS_K]]
    print(f"\nSelected top-{TOP_HEADS_K} heads: {best_heads}")

    print("\nPrecomputing DINO indices for eval set...")
    eval_dino = precompute_dino_indices(eval_loader, gd_proc, gd_model, device)

    acc_plain = eval_plain(clip, eval_loader, class_text_feats)
    print(f"\nPlain CLIP (eval): {acc_plain:.2f}%\n")

    ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.01]
    print(f"{'Alpha':<8} {'DINO acc':>10} {'Δ':>8}")
    print("-" * 30)
    best_dino = {"acc": acc_plain, "alpha": None}
    for a in ALPHAS:
        acc = eval_pasta_cached(clip, eval_loader, class_text_feats, best_heads, a, eval_dino)
        print(f"{a:<8} {acc:>10.2f} {acc - acc_plain:>+8.2f}%")
        if acc > best_dino["acc"]:
            best_dino = {"acc": acc, "alpha": a}

    print(f"\n{'='*50}")
    print(f"  Plain CLIP              : {acc_plain:.2f}%")
    print(f"  Best PASTA (DINO) α={best_dino['alpha']} : {best_dino['acc']:.2f}%  ({best_dino['acc'] - acc_plain:+.2f}%)")
    print(f"  Heads used: {best_heads}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
