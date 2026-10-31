"""Make Figures 2 and 3 of the report from a trained run.

Figure 2 is a 4x4 grid of samples from the prior. Figure 3 shows four test images, their
reconstructions and four new samples. Images are enlarged with nearest-neighbour
interpolation, so the PDF shows the model's real 64x64 pixels without extra smoothing.

Example:
    python make_figures.py --run runs/vae_64_b0.01 --out report/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from data import get_dataset
from vae import VAE


def upscale(images, factor):
    return F.interpolate(images, scale_factor=factor, mode="nearest")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="runs/vae_64_b0.01")
    parser.add_argument("--out", type=str, default="report/figures")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    run_dir, out_dir = Path(args.run), Path(args.out)
    config = json.loads((run_dir / "config.json").read_text())
    device = torch.device("cpu")  # only a handful of images, so the CPU is plenty
    torch.manual_seed(args.seed)

    model = VAE(config["img_size"], config["latent_dim"]).to(device)
    model.load_state_dict(torch.load(run_dir / "vae.pt", map_location=device))
    model.eval()

    with torch.no_grad():
        # Figure 2: 4x4 grid of samples from the prior
        samples = model.sample(16, device).cpu()
        save_image(make_grid(upscale(samples, args.scale), nrow=4, padding=4, pad_value=1.0),
                   out_dir / "samples16_final.png")

        # Figure 3: random test images, their reconstructions (decoding the posterior mean)
        # and new samples from the prior
        test_images = get_dataset("test", config["img_size"]).tensors[0]
        real = test_images[torch.randperm(len(test_images))[:4]]
        mu, _ = model.encode(real.to(device))
        reconstructed = model.decode(mu).cpu()
        generated = model.sample(4, device).cpu()

    rows = [("Real (test)", real), ("Reconstructed", reconstructed), ("Generated", generated)]
    fig, axes = plt.subplots(3, 4, figsize=(6, 4.8))
    for row_axes, (label, images) in zip(axes, rows):
        for ax, image in zip(row_axes, images):
            ax.imshow(image[0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
        row_axes[0].set_ylabel(label, fontsize=11)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.2)
    fig.savefig(out_dir / "real_recon_gen.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
