"""Train a VAE on the chest X-ray training split.

The test split is scored after every epoch. Every `--sample-every` epochs (and at the
end) the script saves the weights, the loss history, a loss plot, and grids of samples
and test reconstructions into `--out`.

Example (the final model in the report):
    python train.py --beta 0.01 --out runs/vae_64_b0.01
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import get_dataset
from vae import VAE, vae_loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--out", type=str, default="runs/vae")
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, beta, optimizer=None):
    """Make one pass over `loader`.

    Trains if an optimizer is given and only evaluates otherwise. Returns each loss
    averaged per image.
    """
    if optimizer is not None:
        is_training = True
        model.train()
    else:
        is_training = False
        model.eval()
    loss_sums = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    num_images = 0
    with torch.set_grad_enabled(is_training):
        for images, _labels in loader:
            images = images.to(device)
            reconstruction, mu, logvar = model(images)
            total_loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, images, mu, logvar, beta)
            if is_training:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            batch_size = images.size(0)
            loss_sums["loss"] += total_loss.item() * batch_size
            loss_sums["recon"] += reconstruction_loss.item() * batch_size
            loss_sums["kl"] += kl_loss.item() * batch_size
            num_images += batch_size
    return {name: total / num_images for name, total in loss_sums.items()}


def plot_losses(history, path):
    """Plot train and test curves for the total loss, BCE and KL side by side."""
    epochs = range(1, len(history["train"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    panels = [("loss", "Total loss (neg. ELBO)"), ("recon", "Reconstruction (BCE)"), ("kl", "KL divergence")]
    for ax, (loss_name, title) in zip(axes, panels):
        ax.plot(epochs, [epoch_losses[loss_name] for epoch_losses in history["train"]], label="train")
        ax.plot(epochs, [epoch_losses[loss_name] for epoch_losses in history["test"]], label="test")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = get_device()
    out_dir = Path(args.out)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_set = get_dataset("train", args.img_size)
    test_set = get_dataset("test", args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256)
    print(f"device={device}  train={len(train_set)}  test={len(test_set)}")

    model = VAE(args.img_size, args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"params: {num_params / 1e6:.2f}M")

    # Reuse the same latent vectors and test images every time so the saved grids can be
    # compared across epochs.
    fixed_z = torch.randn(64, args.latent_dim, device=device)
    fixed_test_images = test_set.tensors[0][:32].to(device)
    history = {"train": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_losses = run_epoch(model, train_loader, device, args.beta, optimizer)
        test_losses = run_epoch(model, test_loader, device, args.beta)
        history["train"].append(train_losses)
        history["test"].append(test_losses)
        print(f"epoch {epoch:3d} | train loss {train_losses['loss']:.1f} "
              f"(rec {train_losses['recon']:.1f}, kl {train_losses['kl']:.1f})"
              f" | test loss {test_losses['loss']:.1f} | {time.time() - start_time:.1f}s")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                generated = model.decode(fixed_z)
                save_image(generated, out_dir / "samples" / f"samples_{epoch:03d}.png", nrow=8)
                reconstruction, _, _ = model(fixed_test_images)
                save_image(torch.cat([fixed_test_images, reconstruction]),
                           out_dir / "samples" / f"recon_{epoch:03d}.png", nrow=8)
            torch.save(model.state_dict(), out_dir / "vae.pt")
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f)
            plot_losses(history, out_dir / "loss_curves.png")


if __name__ == "__main__":
    main()
