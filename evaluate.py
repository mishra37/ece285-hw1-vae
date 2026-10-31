"""Score a trained VAE with FID and Inception Score (IS).

- FID uses pytorch-fid's InceptionV3 (the standard FID weights) on its 2048-d pool
  features.
- IS uses torchvision's ImageNet InceptionV3 with 10 splits.

Three sets of images are scored: samples from the prior, reconstructions of the 5216
training images, and, as a reference for the best FID we could hope for, one random half
of the real training set against the other half (2608 vs 2608). Everything is compared
against the real training images, because the 624-image test set is too small to
estimate a 2048x2048 covariance.

The Inception passes (the slow part) run on the GPU if there is one. The FID statistics
are computed in float64 on the CPU, since MPS doesn't support float64.

Results are written to <run>/metrics.json, along with an 8x8 grid of samples.

Example:
    python evaluate.py --run runs/vae_64_b0.01
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.utils import save_image
from tqdm import tqdm

from data import get_dataset
from train import get_device
from vae import VAE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def extract(images, fid_net, is_net, device, batch_size):
    """Run images (N, 1, H, W) in [0, 1] through both Inception networks.

    Returns the FID pool features (N, 2048) and the IS class probabilities (N, 1000).
    """
    features, probs = [], []
    for start in tqdm(range(0, len(images), batch_size), leave=False):
        batch = images[start:start + batch_size].to(device).repeat(1, 3, 1, 1)  # Inception wants 3 channels
        features.append(fid_net(batch)[0][:, :, 0, 0].cpu().double())  # pytorch-fid resizes to 299x299 itself
        resized = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
        normalized = (resized - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)
        probs.append(F.softmax(is_net(normalized), dim=1).cpu().double())
    return torch.cat(features), torch.cat(probs)


def fid_from_features(features_a, features_b):
    """Frechet distance between Gaussians fitted to two sets of features."""
    a, b = features_a.numpy(), features_b.numpy()
    return float(calculate_frechet_distance(a.mean(0), np.cov(a, rowvar=False),
                                            b.mean(0), np.cov(b, rowvar=False)))


def inception_score(probs, splits=10):
    """exp(mean KL(p(y|x) || p(y))), computed on each split and then averaged."""
    scores = []
    for chunk in probs.chunk(splits):
        marginal = chunk.mean(0, keepdim=True)
        kl = (chunk * (torch.log(chunk + 1e-12) - torch.log(marginal + 1e-12))).sum(1).mean()
        scores.append(torch.exp(kl).item())
    return float(np.mean(scores))


@torch.no_grad()
def generate(model, num_images, device, batch_size, reconstruct_from=None):
    """Make `num_images` images: prior samples, or reconstructions of `reconstruct_from` if given."""
    outputs = []
    for start in range(0, num_images, batch_size):
        if reconstruct_from is None:
            outputs.append(model.sample(min(batch_size, num_images - start), device).cpu())
        else:
            outputs.append(model(reconstruct_from[start:start + batch_size].to(device))[0].cpu())
    return torch.cat(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="runs/vae_64")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run)
    config = json.loads((run_dir / "config.json").read_text())
    device = get_device()
    torch.manual_seed(args.seed)
    print(f"device={device}")

    model = VAE(config["img_size"], config["latent_dim"]).to(device)
    model.load_state_dict(torch.load(run_dir / "vae.pt", map_location=device))
    model.eval()

    fid_net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
    is_net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device).eval()

    train_images = get_dataset("train", config["img_size"]).tensors[0]
    num_images = len(train_images)

    def features_of(images):
        return extract(images, fid_net, is_net, device, args.batch_size)

    print("extracting real train features ...")
    real_features, real_probs = features_of(train_images)

    results = {}
    print("scoring VAE samples ...")
    features, probs = features_of(generate(model, num_images, device, args.batch_size))
    results["vae_samples"] = {"fid": fid_from_features(features, real_features), "is": inception_score(probs)}

    print("scoring VAE reconstructions ...")
    features, probs = features_of(generate(model, num_images, device, args.batch_size, train_images))
    results["vae_reconstructions"] = {"fid": fid_from_features(features, real_features),
                                      "is": inception_score(probs)}

    # Reference: one random half of the real images against the other half.
    order = torch.randperm(num_images)
    half = num_images // 2
    first, second = order[:half], order[half:2 * half]
    results["real_train_half"] = {"fid": fid_from_features(real_features[first], real_features[second]),
                                  "is": inception_score(real_probs[first])}

    for name, scores in results.items():
        print(f"{name:22s} FID {scores['fid']:8.2f}   IS {scores['is']:.3f}")
    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    save_image(model.sample(64, device), run_dir / "generated_grid.png", nrow=8, padding=2)


if __name__ == "__main__":
    main()
