# Chest X-ray VAE (ECE 285, Homework 1)

A convolutional variational autoencoder trained on the Kaggle chest X-ray pneumonia
dataset at 64x64, with a sweep over the KL weight β. The write-up is in
[`report/main.pdf`](report/main.pdf).

## Files

- `data.py`: loads and caches the dataset (grayscale, 64x64, values in [0, 1])
- `vae.py`: the VAE model
- `train.py`: training loop
- `evaluate.py`: FID and Inception Score
- `make_figures.py`: figures for the report
- `runs/`: configs, loss histories, metrics and sample images (checkpoints not included)

## Usage

Download the dataset from Kaggle (`paultimothymooney/chest-xray-pneumonia`) and place it at
`data/chest_xray/{train,test}/{NORMAL,PNEUMONIA}/`.

```bash
pip install -r requirements.txt
python train.py --beta 0.01 --out runs/vae_64_b0.01   # final model
python evaluate.py --run runs/vae_64_b0.01
```

Other runs in the report: `--beta` 1, 0.1, 0.05, 0.03, and `--beta 0.03 --latent-dim 256`.
All runs use the defaults: 100 epochs, batch size 64, Adam with lr 1e-3, seed 0.
