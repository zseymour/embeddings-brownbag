# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy",
#     "pillow",
#     "scipy",
#     "torch",
#     "transformers",
#     "matplotlib",
# ]
# ///
"""DINOv2 (2023) patch features vs. hand-wired 2018 semantic segmentation, scene-appropriate recipe.

Reproduces the *spirit* of the DINOv2 paper's Figure 1 (unsupervised structure falls out of
frozen patch features) on three of the talk author's own 2018 place-recognition query frames,
next to the hand-built semantic-segmentation panel from the same 2018 figure - but street
scenes are not the object-centric photos DINOv2's own recipe was built for, so this version
drops the paper's foreground/background PCA threshold (it assumes one salient object on a
background, which is wrong for a road scene) and instead reports two things per frame:

  (a) a per-frame 3-component PCA of ALL patches, mapped to RGB - the closest unsupervised
      analogue of "does structure fall out", with no masking;
  (b) a k=6 k-means over L2-normalised patch features, fit JOINTLY across all three frames so
      the same cluster id gets the same colour in every frame - the fair visual comparison to
      a hand-wired segmentation map, plus an honest purity number against the 2018 labels.

Crops come from the top row of `saane-retrieval-nordland.png`, `saane-retrieval-robotcar.png`
and `saane-supp-4.png`: each is a 5-column, 2-row grid of ~600px-square panels on a white
background. Panels are found by connected-component analysis on non-white pixels (this also
rejects the small rotated row/column text labels, which are far smaller than a panel). Column 1
is the query photo, column 3 is the semantic-segmentation map.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = Path(__file__).resolve().parent
FIGURES_DIR = HERE.parent / "public" / "figures"
RESULTS_DIR = HERE / "results"
CROPS_DIR = RESULTS_DIR / "e9_crops"

# Model/dataset caches must live under the experiment dir, not the user's home dir.
os.environ.setdefault("HF_HOME", str(HERE / ".cache"))
os.environ.setdefault("TORCH_HOME", str(HERE / ".cache"))

import torch  # noqa: E402  (import after HF_HOME/TORCH_HOME are set)
from transformers import AutoModel  # noqa: E402

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

BG = "#12151c"
FG = "#e8ecf1"
MUTED = "#98a2b3"

SOURCES = {
    "nordland": "saane-retrieval-nordland.png",
    "robotcar": "saane-retrieval-robotcar.png",
    "supp4": "saane-supp-4.png",
}
ROW_LABELS = {"nordland": "Nordland", "robotcar": "RobotCar", "supp4": "Suburban"}

WHITE_THRESHOLD = 245  # a pixel counts as "ink" if any channel is below this
MIN_PANEL_AREA = 20_000  # px; separates ~600x600 photo/map panels from small text labels

MODEL_NAME = "facebook/dinov2-base"  # 0.79s/image at 896x896 on an M4 - dinov2-small is not needed
PATCH_SIZE = 14
DINO_INPUT = 896  # 896 / 14 = 64, a fine-enough patch grid for road/sky/building/vegetation structure
GRID = DINO_INPUT // PATCH_SIZE  # 64
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

STANDALONE_SIZE = GRID * 25  # 1600px, an exact integer nearest-neighbour scale from the 64x64 grid

N_CLUSTERS = 6
KMEANS_SEED = 0
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 200
# Fixed palette, one colour per cluster id, shared across all three frames.
CLUSTER_PALETTE = [
    (124, 196, 255),  # blue
    (232, 137, 74),  # orange
    (52, 211, 153),  # emerald
    (242, 214, 90),  # yellow
    (199, 146, 234),  # violet
    (255, 107, 129),  # rose
]

SEG_CLASS_MERGE_DIST = 40  # RGB Euclidean distance below which two segmentation colours are the same class
MAX_SEG_CLASSES = 12  # cap on distinct 2018-segmentation classes; long-tail anti-aliasing colours fold into these


# --- Step 1: locate and crop panels -----------------------------------------------------------


def find_panels(path: Path) -> tuple[tuple[int, int], list[dict]]:
    """Connected components of non-white pixels, area-filtered to panel-sized blobs."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    non_white = np.any(arr < WHITE_THRESHOLD, axis=-1)
    labeled, n = ndimage.label(non_white)
    boxes = []
    for i in range(1, n + 1):
        ys, xs = np.where(labeled == i)
        area = int(ys.size)
        if area < MIN_PANEL_AREA:
            continue
        boxes.append({"bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], "area": area})
    return img.size, boxes


def cluster_columns(boxes: list[dict], tol: int = 60) -> list[tuple[int, list[dict]]]:
    """Group panel boxes into grid columns by x0, tolerant of a few px of row-to-row jitter."""
    cols: list[tuple[int, list[dict]]] = []
    for b in sorted(boxes, key=lambda b: b["bbox"][0]):
        x0 = b["bbox"][0]
        for xc, members in cols:
            if abs(xc - x0) < tol:
                members.append(b)
                break
        else:
            cols.append((x0, [b]))
    cols.sort(key=lambda c: c[0])
    return cols


def crop_query_and_semseg(source_name: str) -> tuple[Image.Image, Image.Image, list[int], list[int]]:
    """Top-row column 1 (query) and column 3 (semantic segmentation) of a SAANE retrieval figure."""
    path = FIGURES_DIR / source_name
    img = Image.open(path).convert("RGB")
    _, boxes = find_panels(path)
    cols = cluster_columns(boxes)
    if len(cols) != 5:
        raise RuntimeError(f"{source_name}: expected 5 panel columns, found {len(cols)}")
    query_box = min(cols[0][1], key=lambda b: b["bbox"][1])["bbox"]
    semseg_box = min(cols[2][1], key=lambda b: b["bbox"][1])["bbox"]
    query = img.crop((query_box[0], query_box[1], query_box[2] + 1, query_box[3] + 1))
    semseg = img.crop((semseg_box[0], semseg_box[1], semseg_box[2] + 1, semseg_box[3] + 1))
    return query, semseg, query_box, semseg_box


# --- Step 2: DINOv2 patch features ----------------------------------------------------------


def load_dinov2(device: str):
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval().to(device)
    return model


def patch_features(model, device: str, crop: Image.Image) -> np.ndarray:
    """Resize to DINO_INPUT x DINO_INPUT, run DINOv2, return the (GRID*GRID, hidden) patch tokens."""
    resized = crop.resize((DINO_INPUT, DINO_INPUT), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pixel_values=tensor, interpolate_pos_encoding=True)
    tokens = out.last_hidden_state[0, 1:, :]  # drop CLS token
    return tokens.cpu().numpy().astype(np.float64)


# --- Step 3a: per-frame PCA -------------------------------------------------------------------


def pca_fit_transform(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-center and SVD. Returns (scores[N, n_components], components[n_components, D],
    explained_variance_ratio for every retained singular value, i.e. the full spectrum)."""
    mean = X.mean(axis=0)
    centered = X - mean
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    explained_variance = (s**2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / explained_variance.sum()
    components = vt[:n_components]
    scores = centered @ components.T
    return scores, components, explained_variance_ratio


def min_max_normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def per_frame_pca_rgb(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """3-component PCA over one frame's own (GRID*GRID, hidden) patches, min-max to RGB. No masking."""
    scores, _components, evr = pca_fit_transform(features, n_components=3)
    rgb = np.stack([min_max_normalise(scores[:, c]) for c in range(3)], axis=1)
    return rgb.reshape(GRID, GRID, 3), evr[:3]


# --- Step 3b: joint k-means on L2-normalised features ---------------------------------------


def kmeans_plus_plus_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    centroids = np.empty((k, x.shape[1]), dtype=x.dtype)
    first = rng.integers(n)
    centroids[0] = x[first]
    closest_d2 = np.sum((x - centroids[0]) ** 2, axis=1)
    for i in range(1, k):
        total = closest_d2.sum()
        probs = closest_d2 / total if total > 0 else np.full(n, 1.0 / n)
        idx = rng.choice(n, p=probs)
        centroids[i] = x[idx]
        d2 = np.sum((x - centroids[i]) ** 2, axis=1)
        closest_d2 = np.minimum(closest_d2, d2)
    return centroids


def kmeans(x: np.ndarray, k: int, seed: int, n_init: int, max_iter: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Deterministic (seeded) Lloyd's k-means with k-means++ init and multiple restarts;
    keeps the lowest-inertia run. Returns (labels, centroids, inertia)."""
    rng = np.random.default_rng(seed)
    x_sq = np.sum(x**2, axis=1)
    best_inertia = np.inf
    best_labels = None
    best_centroids = None
    for _ in range(n_init):
        centroids = kmeans_plus_plus_init(x, k, rng)
        labels = None
        for _it in range(max_iter):
            d2 = x_sq[:, None] - 2.0 * x @ centroids.T + np.sum(centroids**2, axis=1)[None, :]
            new_labels = np.argmin(d2, axis=1)
            new_centroids = centroids.copy()
            for j in range(k):
                members = x[new_labels == j]
                if members.shape[0] > 0:
                    new_centroids[j] = members.mean(axis=0)
            shift = float(np.sum((new_centroids - centroids) ** 2))
            centroids = new_centroids
            if labels is not None and np.array_equal(labels, new_labels):
                labels = new_labels
                break
            labels = new_labels
            if shift < 1e-10:
                break
        d2 = x_sq[:, None] - 2.0 * x @ centroids.T + np.sum(centroids**2, axis=1)[None, :]
        inertia = float(np.sum(d2[np.arange(x.shape[0]), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    return best_labels, best_centroids, best_inertia


def cluster_rgb_grid(labels_for_frame: np.ndarray) -> np.ndarray:
    palette = np.array(CLUSTER_PALETTE, dtype=np.float64) / 255.0
    rgb = palette[labels_for_frame]
    return rgb.reshape(GRID, GRID, 3)


# --- Step 4: 2018 segmentation classes and cluster purity -----------------------------------


def global_segmentation_classes(
    semseg_crops: dict[str, Image.Image],
) -> tuple[dict[str, np.ndarray], list[tuple[int, int, int]]]:
    """Resize every frame's segmentation panel to GRID x GRID (nearest), then build ONE shared
    palette of classes across all three frames by merging near-identical colours (anti-aliasing
    noise) into the nearest frequent colour. Returns per-frame (GRID, GRID) class-id grids and
    the canonical RGB colour of each class id."""
    per_frame_pixels = {}
    combined_counts: dict[tuple[int, int, int], int] = {}
    for key, img in semseg_crops.items():
        small = img.resize((GRID, GRID), Image.NEAREST)
        arr = np.asarray(small).reshape(-1, 3)
        per_frame_pixels[key] = arr
        colors, counts = np.unique(arr, axis=0, return_counts=True)
        for color, count in zip(colors, counts):
            key_t = (int(color[0]), int(color[1]), int(color[2]))
            combined_counts[key_t] = combined_counts.get(key_t, 0) + int(count)

    sorted_colors = sorted(combined_counts.items(), key=lambda kv: -kv[1])
    canonical: list[np.ndarray] = []
    color_to_class: dict[tuple[int, int, int], int] = {}
    for color, _count in sorted_colors:
        c_arr = np.array(color, dtype=np.float64)
        assigned = None
        for idx, canon_rgb in enumerate(canonical):
            if np.sum((canon_rgb - c_arr) ** 2) <= SEG_CLASS_MERGE_DIST**2:
                assigned = idx
                break
        if assigned is None:
            if len(canonical) < MAX_SEG_CLASSES:
                canonical.append(c_arr)
                assigned = len(canonical) - 1
            else:
                dists = [float(np.sum((canon_rgb - c_arr) ** 2)) for canon_rgb in canonical]
                assigned = int(np.argmin(dists))
        color_to_class[color] = assigned

    class_grids = {}
    for key, arr in per_frame_pixels.items():
        ids = np.array([color_to_class[(int(p[0]), int(p[1]), int(p[2]))] for p in arr], dtype=np.int64)
        class_grids[key] = ids.reshape(GRID, GRID)
    canonical_colors = [(int(c[0]), int(c[1]), int(c[2])) for c in canonical]
    return class_grids, canonical_colors


def cluster_purity(cluster_labels: np.ndarray, class_labels: np.ndarray, n_clusters: int, grid: int) -> dict:
    """Classic purity metric restricted to one frame: for each of the n_clusters k-means
    clusters, the majority 2018-segmentation class among that frame's members, the fraction
    matching it, and where the cluster sits in the frame (mean patch row/col, row 0 = top);
    plus the frame-level size-weighted purity."""
    n = cluster_labels.shape[0]
    row_idx = np.repeat(np.arange(grid), grid)
    col_idx = np.tile(np.arange(grid), grid)
    per_cluster = []
    correct_total = 0
    for c in range(n_clusters):
        mask = cluster_labels == c
        size = int(mask.sum())
        if size == 0:
            per_cluster.append(
                {"cluster": c, "size": 0, "majority_class": None, "purity": None, "mean_row": None, "mean_col": None}
            )
            continue
        classes_in_cluster = class_labels[mask]
        vals, counts = np.unique(classes_in_cluster, return_counts=True)
        majority_idx = int(np.argmax(counts))
        majority_class = int(vals[majority_idx])
        majority_count = int(counts[majority_idx])
        correct_total += majority_count
        per_cluster.append(
            {
                "cluster": c,
                "size": size,
                "majority_class": majority_class,
                "purity": majority_count / size,
                "mean_row": float(row_idx[mask].mean()),
                "mean_col": float(col_idx[mask].mean()),
            }
        )
    return {"per_cluster": per_cluster, "frame_purity": correct_total / n}


def upsample_nn(grid_rgb: np.ndarray, size) -> np.ndarray:
    """Nearest-neighbour upsample a (GRID, GRID, 3) float [0,1] array to `size` (w, h) pixels."""
    if isinstance(size, int):
        size = (size, size)
    img = Image.fromarray((np.clip(grid_rgb, 0.0, 1.0) * 255.0).astype(np.uint8))
    return np.asarray(img.resize(size, Image.NEAREST))


# --- Main pipeline -----------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Step 1: crop, and save crops for visual verification.
    crops: dict[str, dict] = {}
    for key, source_name in SOURCES.items():
        query, semseg, query_box, semseg_box = crop_query_and_semseg(source_name)
        query.save(CROPS_DIR / f"{key}-query.png")
        semseg.save(CROPS_DIR / f"{key}-semseg.png")
        crops[key] = {
            "source": source_name,
            "query_bbox_xyxy": query_box,
            "semseg_bbox_xyxy": semseg_box,
            "query_crop_size": list(query.size),
            "semseg_crop_size": list(semseg.size),
            "query_img": query,
            "semseg_img": semseg,
        }

    # Step 2: DINOv2 patch features, one (GRID*GRID, hidden) block per frame.
    model = load_dinov2(device)
    hidden_dim = model.config.hidden_size
    feature_blocks = []
    for key in SOURCES:
        feats = patch_features(model, device, crops[key]["query_img"])
        crops[key]["features"] = feats  # (4096, hidden_dim)
        feature_blocks.append(feats)
    n_per_image = GRID * GRID

    # Step 3a: per-frame PCA (no joint fit, no threshold, all patches).
    per_frame_pca = {}
    per_frame_evr = {}
    for key in SOURCES:
        rgb_grid, evr = per_frame_pca_rgb(crops[key]["features"])
        per_frame_pca[key] = rgb_grid
        per_frame_evr[key] = evr

    # Step 3b: k-means (k=6) on L2-normalised patch features, fit jointly across all 3 frames.
    all_features = np.concatenate(feature_blocks, axis=0)  # (3*4096, hidden_dim)
    norms = np.linalg.norm(all_features, axis=1, keepdims=True)
    all_features_l2 = all_features / np.clip(norms, 1e-12, None)
    labels, _centroids, inertia = kmeans(
        all_features_l2, N_CLUSTERS, seed=KMEANS_SEED, n_init=KMEANS_N_INIT, max_iter=KMEANS_MAX_ITER
    )
    per_frame_labels = {}
    per_frame_kmeans_rgb = {}
    for idx, key in enumerate(SOURCES):
        frame_labels = labels[idx * n_per_image : (idx + 1) * n_per_image]
        per_frame_labels[key] = frame_labels
        per_frame_kmeans_rgb[key] = cluster_rgb_grid(frame_labels)

    # Step 4: 2018 segmentation classes (shared palette across frames) and cluster purity.
    semseg_imgs = {key: crops[key]["semseg_img"] for key in SOURCES}
    class_grids, canonical_seg_colors = global_segmentation_classes(semseg_imgs)
    purity_by_frame = {}
    for key in SOURCES:
        class_flat = class_grids[key].reshape(-1)
        purity_by_frame[key] = cluster_purity(per_frame_labels[key], class_flat, N_CLUSTERS, GRID)

    # --- Figures ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG}
    )

    def style_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Composite 3x4 figure: rows = frames, columns = query / 2018 semseg / DINOv2 PCA / DINOv2 k-means.
    fig = plt.figure(figsize=(20, 16.2), dpi=150)
    gs = fig.add_gridspec(3, 4, hspace=0.10, wspace=0.04, left=0.035, right=0.99, top=0.93, bottom=0.02)
    col_titles = [
        "Query frame",
        "2018 semantic segmentation (hand-wired)",
        "DINOv2 PCA, per-frame (no supervision)",
        "DINOv2 k-means, k=6, joint fit (no supervision)",
    ]

    standalone_paths = []
    for row, key in enumerate(SOURCES):
        crop_size = crops[key]["query_crop_size"]
        pca_upsampled = upsample_nn(per_frame_pca[key], tuple(crop_size))
        kmeans_upsampled = upsample_nn(per_frame_kmeans_rgb[key], tuple(crop_size))

        panels = [
            np.asarray(crops[key]["query_img"]),
            np.asarray(crops[key]["semseg_img"]),
            pca_upsampled,
            kmeans_upsampled,
        ]
        for col, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(panel)
            style_axis(ax)
            if row == 0:
                ax.set_title(col_titles[col], color=MUTED, fontsize=15, pad=10)
            if col == 0:
                ax.set_ylabel(ROW_LABELS[key], color=MUTED, fontsize=15)

        # Standalone per-frame PNGs, upsampled straight from the 64x64 grid, no text.
        pca_standalone = upsample_nn(per_frame_pca[key], STANDALONE_SIZE)
        pca_path = FIGURES_DIR / f"exp-e9-dino-pca-{key}.png"
        Image.fromarray(pca_standalone).save(pca_path)
        standalone_paths.append(str(pca_path.relative_to(FIGURES_DIR.parent.parent)))

        kmeans_standalone = upsample_nn(per_frame_kmeans_rgb[key], STANDALONE_SIZE)
        kmeans_path = FIGURES_DIR / f"exp-e9-dino-pca-{key}-kmeans.png"
        Image.fromarray(kmeans_standalone).save(kmeans_path)
        standalone_paths.append(str(kmeans_path.relative_to(FIGURES_DIR.parent.parent)))

    composite_path = FIGURES_DIR / "exp-e9-dino-pca.png"
    fig.savefig(composite_path, facecolor=BG)
    plt.close(fig)

    # --- Results JSON ---
    results = {
        "model": MODEL_NAME,
        "hidden_dim": int(hidden_dim),
        "device": device,
        "seed": SEED,
        "dino_input_size": DINO_INPUT,
        "patch_size": PATCH_SIZE,
        "patch_grid": [GRID, GRID],
        "crops": {
            key: {
                "source": v["source"],
                "query_bbox_xyxy": v["query_bbox_xyxy"],
                "semseg_bbox_xyxy": v["semseg_bbox_xyxy"],
                "query_crop_size": v["query_crop_size"],
                "semseg_crop_size": v["semseg_crop_size"],
            }
            for key, v in crops.items()
        },
        "panel_detection": {
            "method": "connected components on non-white pixels (any RGB channel < threshold)",
            "white_threshold": WHITE_THRESHOLD,
            "min_panel_area_px": MIN_PANEL_AREA,
        },
        "pca_per_frame": {
            "description": (
                "3-component PCA fit independently on each frame's own patch features (no cross-frame "
                "join, no foreground/background threshold - every patch is shown)."
            ),
            "explained_variance_ratio_pc1_pc2_pc3": {
                key: [float(v) for v in per_frame_evr[key]] for key in SOURCES
            },
            "normalisation": "per-component min-max to [0, 1] over that frame's own patches",
        },
        "kmeans_joint": {
            "description": (
                "k-means (k=6) on L2-normalised patch features, fit jointly over all 3 frames' patches "
                "so a cluster id has the same fixed colour in every frame."
            ),
            "n_clusters": N_CLUSTERS,
            "seed": KMEANS_SEED,
            "n_init": KMEANS_N_INIT,
            "max_iter": KMEANS_MAX_ITER,
            "inertia": inertia,
            "palette_rgb": CLUSTER_PALETTE,
        },
        "segmentation_classes": {
            "description": (
                "2018 segmentation panels resized to the GRID x GRID patch grid (nearest neighbour); "
                "colours merged across all 3 frames (Euclidean RGB distance <= "
                f"{SEG_CLASS_MERGE_DIST}) into one shared class palette."
            ),
            "merge_distance": SEG_CLASS_MERGE_DIST,
            "n_classes": len(canonical_seg_colors),
            "canonical_class_rgb": canonical_seg_colors,
        },
        "cluster_purity_vs_2018_segmentation": purity_by_frame,
        "figures": {
            "composite": str(composite_path.relative_to(FIGURES_DIR.parent.parent)),
            "standalone": standalone_paths,
        },
    }

    results_path = RESULTS_DIR / "e9_dino_pca.json"
    results_path.write_text(json.dumps(results, indent=2))

    print(f"Wrote {results_path}")
    print(f"Wrote {composite_path}")
    for p in standalone_paths:
        print(f"Wrote {FIGURES_DIR.parent.parent / p}")
    print("segmentation classes:", canonical_seg_colors)
    for key in SOURCES:
        print(key, "frame_purity", purity_by_frame[key]["frame_purity"])
        for row in purity_by_frame[key]["per_cluster"]:
            print("   ", row)


if __name__ == "__main__":
    main()
