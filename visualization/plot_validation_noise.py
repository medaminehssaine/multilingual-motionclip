import numpy as np
import matplotlib.pyplot as plt

noise = np.array([0.0, 0.1, 0.25, 0.5, 1.0])

# Validation results: Original KL=1e-3
original_mean = np.array([
    0.702731,
    0.697159,
    0.704900,
    0.824304,
    1.173424,
])

original_std = np.array([
    0.801671,
    0.770377,
    0.708990,
    0.711439,
    0.775619,
])

# Validation results: KL=1e-3 + beta_train=0.10 fine-tuning
finetuned_mean = np.array([
    0.657331,
    0.654283,
    0.660558,
    0.699445,
    0.874478,
])

finetuned_std = np.array([
    0.724534,
    0.718593,
    0.708674,
    0.679836,
    0.664303,
])

fig, ax = plt.subplots(figsize=(8, 5.5))

# Original model
ax.plot(
    noise,
    original_mean,
    marker="o",
    linewidth=2,
    label="Original KL=1e-3",
)
ax.fill_between(
    noise,
    original_mean - original_std,
    original_mean + original_std,
    alpha=0.15,
)

# Fine-tuned model
ax.plot(
    noise,
    finetuned_mean,
    marker="o",
    linewidth=2,
    label=r"Fine-tuned $\beta_{\mathrm{train}}=0.10$",
)
ax.fill_between(
    noise,
    finetuned_mean - finetuned_std,
    finetuned_mean + finetuned_std,
    alpha=0.15,
)

ax.set_xlabel("Evaluation noise scale β")
ax.set_ylabel("Reconstruction MSE")
ax.set_title("Validation Reconstruction Robustness to Latent Noise")

ax.set_xticks(noise)
ax.grid(True, alpha=0.25)
ax.legend()

plt.tight_layout()

plt.savefig(
    "validation_noise_robustness.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
