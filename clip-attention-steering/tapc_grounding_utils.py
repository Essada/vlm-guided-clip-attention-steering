def bbox_to_patch_indices_strict(
    bbox,
    orig_w,
    orig_h,
    patch_size,
    grid_size,
    crop_size=224,
    min_patch_overlap=0.5,
):
    x1, y1, x2, y2 = bbox

    scale = crop_size / min(orig_w, orig_h)
    new_w = orig_w * scale
    new_h = orig_h * scale
    x1, y1, x2, y2 = x1 * scale, y1 * scale, x2 * scale, y2 * scale

    left = (new_w - crop_size) / 2
    top = (new_h - crop_size) / 2
    x1 = max(0.0, x1 - left)
    y1 = max(0.0, y1 - top)
    x2 = min(float(crop_size), x2 - left)
    y2 = min(float(crop_size), y2 - top)

    if x2 <= x1 or y2 <= y1:
        return []

    selected = []
    patch_area = patch_size * patch_size
    for r in range(grid_size):
        for c in range(grid_size):
            px1 = c * patch_size
            py1 = r * patch_size
            px2 = px1 + patch_size
            py2 = py1 + patch_size
            ix1 = max(x1, px1)
            iy1 = max(y1, py1)
            ix2 = min(x2, px2)
            iy2 = min(y2, py2)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            overlap_area = (ix2 - ix1) * (iy2 - iy1)
            if overlap_area / patch_area >= min_patch_overlap:
                selected.append(r * grid_size + c)
    return selected
