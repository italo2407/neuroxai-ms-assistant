import base64
import io
import numpy as np
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")


def decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded bytes to numpy float32 grayscale [0,1] array (H, W)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def decode_mask_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode uploaded mask bytes to binary float32 array (H, W)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return (arr > 0.5).astype(np.float32)


def resize_to(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize (H,W) float32 array to (size, size) using bilinear interpolation."""
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Convert (H,W) float32 array [0,1] to base64-encoded grayscale PNG."""
    img_uint8 = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(img_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def colorize_heatmap(arr: np.ndarray, cmap_name: str = "hot") -> str:
    """
    Colorize a float32 (H,W) attribution map using a matplotlib colormap.
    Returns base64-encoded RGBA PNG.
    """
    arr_norm = arr.copy()
    if arr_norm.max() > arr_norm.min():
        arr_norm = (arr_norm - arr_norm.min()) / (arr_norm.max() - arr_norm.min())
    cmap = matplotlib.colormaps[cmap_name]
    rgba = cmap(arr_norm)  # (H, W, 4) float64
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_mask_on_image(image_np: np.ndarray, mask_np: np.ndarray,
                           color: tuple = (255, 80, 80),
                           alpha: float = 0.72,
                           bg_dim: float = 0.45) -> str:
    """
    Overlay a binary mask on a grayscale MRI image with high visibility.
    Mask pixels are blended with `color` at `alpha`.
    Non-mask (background) pixels are dimmed by `bg_dim` to maximise contrast.
    Returns base64 RGB PNG.
    """
    rgb = np.stack([image_np * 255] * 3, axis=-1).astype(np.float32)
    mask_bool = mask_np > 0.5
    out = rgb.copy()
    out[~mask_bool] *= bg_dim
    out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * np.array(color, dtype=np.float32)
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_fpfn_on_image(image_np: np.ndarray, pred_np: np.ndarray,
                           gt_np: np.ndarray,
                           alpha: float = 0.78,
                           bg_dim: float = 0.40) -> str:
    """
    Genera overlay de FP/FN/TP sobre la imagen MRI con alta visibilidad:
      - TP  → verde  (80, 210, 80)
      - FP  → rojo   (230, 50,  50)
      - FN  → azul   (50,  130, 235)
      - TN  → fondo dimmed para maximizar contraste
    Devuelve base64 RGB PNG.
    """
    pred_bin = pred_np > 0.5
    gt_bin   = gt_np  > 0.5

    tp = pred_bin &  gt_bin
    fp = pred_bin & ~gt_bin
    fn = ~pred_bin &  gt_bin
    tn = ~pred_bin & ~gt_bin

    rgb = np.stack([image_np * 255] * 3, axis=-1).astype(np.float32)
    out = rgb.copy()

    # Dim true-negative background to increase salience of colored regions
    out[tn] *= bg_dim

    for mask, color in [
        (tp, np.array([80,  210,  80])),
        (fp, np.array([230,  50,  50])),
        (fn, np.array([50,  130, 235])),
    ]:
        if mask.any():
            out[mask] = (1 - alpha) * out[mask] + alpha * color

    img = Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_heatmap_on_image(image_np: np.ndarray, attr_map: np.ndarray,
                             cmap_name: str = "hot",
                             alpha_min: float = 0.28,
                             alpha_max: float = 0.80) -> str:
    """
    Blend a normalized attribution heatmap over a grayscale MRI image.
    Uses a per-pixel weight in [alpha_min, alpha_max] driven by the activation
    value, so even low-activation pixels show some colour and high-activation
    zones are strongly highlighted.
    Returns base64-encoded RGB PNG.
    """
    arr = attr_map.copy().astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())

    mri_rgb = np.stack([image_np] * 3, axis=-1).astype(np.float32)   # [0,1]
    cmap = matplotlib.colormaps[cmap_name]
    heatmap_rgb = cmap(arr)[:, :, :3].astype(np.float32)              # [0,1]

    weight = (alpha_min + (alpha_max - alpha_min) * arr)[:, :, np.newaxis]
    blended = (1 - weight) * mri_rgb + weight * heatmap_rgb

    img = Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def overlay_heatmap_with_prediction(
    image_np: np.ndarray,
    attr_map: np.ndarray,
    pred_mask_np: np.ndarray,
    cmap_name: str = "hot",
    alpha_min: float = 0.25,
    alpha_max: float = 0.75,
    contour_color: tuple = (50, 230, 50),
    contour_width: int = 2,
) -> str:
    """
    Blend attribution heatmap over the MRI and draw the prediction mask contour.
    contour_color (green by default) outlines the predicted lesion region.
    Returns base64-encoded RGB PNG.
    """
    arr = attr_map.copy().astype(np.float32)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())

    mri_rgb     = np.stack([image_np] * 3, axis=-1).astype(np.float32)
    cmap        = matplotlib.colormaps[cmap_name]
    heatmap_rgb = cmap(arr)[:, :, :3].astype(np.float32)
    weight      = (alpha_min + (alpha_max - alpha_min) * arr)[:, :, np.newaxis]
    blended     = (1 - weight) * mri_rgb + weight * heatmap_rgb  # [0,1]

    # Draw prediction contour using PIL (no extra deps)
    pred_bin = (pred_mask_np > 0.5).astype(np.uint8) * 255
    pred_pil  = Image.fromarray(pred_bin, mode="L")
    eroded    = pred_pil.filter(ImageFilter.MinFilter(contour_width * 2 + 1))
    border    = (np.array(pred_pil) > 0) & ~(np.array(eroded) > 0)
    if border.any():
        c = np.array(contour_color, dtype=np.float32) / 255.0
        blended[border] = c

    img = Image.fromarray((blended * 255).clip(0, 255).astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def base64_to_numpy(b64_str: str) -> np.ndarray:
    """Decode base64 PNG to float32 numpy array."""
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    return np.array(img, dtype=np.float32) / 255.0
