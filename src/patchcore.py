"""PatchCore anomaly detection (Roth et al., 2022), written from the paper.

The model learns only from good parts:
1. A frozen ImageNet CNN gives a feature vector for each small patch of the image.
2. A memory bank keeps a small, diverse subset (coreset) of the good-part patch features.
3. At test time, each patch gets a score: the distance to its nearest neighbour in the memory bank.
   A patch far from all good patches is a defect. The image score is the largest patch score.
"""
import torch
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

from data import CROP_SIZE


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class FeatureExtractor(torch.nn.Module):
    """Mid-level features from layer2 and layer3 of a frozen Wide ResNet-50."""

    def __init__(self):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return B x D x H x W patch features (D = 1536, H = W = 28 for 224 px input)."""
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        # Local neighbourhood average: each patch also sees its 3 x 3 neighbours.
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)


def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Pick indices of a subset that covers the feature space (k-center greedy).

    A random projection makes the distance math faster. It keeps distances close to the originals.
    """
    n = features.shape[0]
    k = max(1, int(n * ratio))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, generator=gen).to(features.device)
    reduced = features @ proj

    selected = torch.empty(k, dtype=torch.long)
    first = int(torch.randint(n, (1,), generator=gen))
    selected[0] = first
    min_dist = torch.cdist(reduced, reduced[first:first + 1]).squeeze(1)
    for i in range(1, k):
        nxt = int(torch.argmax(min_dist))
        selected[i] = nxt
        min_dist = torch.minimum(min_dist, torch.cdist(reduced, reduced[nxt:nxt + 1]).squeeze(1))
    return selected


def nearest_distance(queries: torch.Tensor, bank: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Distance from each query row to its nearest row in the bank."""
    out = []
    for start in range(0, queries.shape[0], chunk):
        d = torch.cdist(queries[start:start + chunk], bank)
        out.append(d.min(dim=1).values)
    return torch.cat(out)


def gaussian_blur(maps: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Smooth B x 1 x H x W anomaly maps."""
    radius = int(4 * sigma + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=maps.dtype, device=maps.device)
    k1 = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    maps = F.conv2d(F.pad(maps, (radius, radius, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1))
    return F.conv2d(F.pad(maps, (0, 0, radius, radius), mode="reflect"), k1.view(1, 1, -1, 1))


class PatchCore:
    def __init__(self, device: torch.device, coreset_ratio: float = 0.1):
        self.device = device
        self.coreset_ratio = coreset_ratio
        self.extractor = FeatureExtractor().to(device)
        self.bank: torch.Tensor | None = None
        self.grid: tuple[int, int] | None = None

    def fit(self, batches) -> int:
        """Build the memory bank from batches of good images. Return the bank size."""
        feats = []
        for x in batches:
            f = self.extractor(x.to(self.device))
            self.grid = (f.shape[2], f.shape[3])
            feats.append(f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]))
        all_feats = torch.cat(feats)
        idx = greedy_coreset(all_feats, self.coreset_ratio)
        self.bank = all_feats[idx.to(all_feats.device)].contiguous()
        return self.bank.shape[0]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (image scores B, anomaly maps B x 1 x 224 x 224) on the CPU."""
        if self.bank is None:
            raise RuntimeError("call fit() before predict()")
        f = self.extractor(x.to(self.device))
        b, d, h, w = f.shape
        dist = nearest_distance(f.permute(0, 2, 3, 1).reshape(-1, d), self.bank)
        patch_maps = dist.view(b, 1, h, w)
        scores = patch_maps.flatten(1).max(dim=1).values
        maps = F.interpolate(patch_maps, size=(CROP_SIZE, CROP_SIZE), mode="bilinear", align_corners=False)
        return scores.cpu(), gaussian_blur(maps).cpu()
