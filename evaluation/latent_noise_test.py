import yaml
import torch
import numpy as np
from os.path import join as pjoin

from src.utils.mclip_wrapper import MClipTextEncoder
from src.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from src.models.modeltype.motionclip import MOTIONCLIP
from src.utils.misc import load_model_wo_clip


DATAPATH = "./data/HumanML3D"
DEVICE = torch.device("cuda:0")

MODELS = [
    {
        "name": "KL=1e-3",
        "ckpt": "./exps/vae-en-mu-kl1e-3-beta05-ft/checkpoint_0025.pth.tar",
        "opt": "./exps/vae-en-mu-kl1e-3/opt.yaml",
    },
    {
        "name": "KL=0",
        "ckpt": "./exps/vae-en-mu-kl0/checkpoint_0020.pth.tar",
        "opt": "./exps/vae-en-mu-kl0/opt.yaml",
    },
]

BETAS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00]

# Same motion for both models.
MOTION_NAME = "000002"


def load_one_model(opt_path, ckpt_path):
    with open(opt_path, "r") as f:
        params = yaml.safe_load(f)

    params["device"] = 0

    clip_model = MClipTextEncoder(device=DEVICE)
    clip_model.eval()

    encoder = Encoder_TRANSFORMER(
        **params,
        njoints=263,
        nfeats=1,
        num_classes=1,
    )

    decoder = Decoder_TRANSFORMER(
        **params,
        njoints=263,
        nfeats=1,
        num_classes=1,
    )

    params["outputxyz"] = "rcxyz" in params["lambdas"]

    model = MOTIONCLIP(
        encoder,
        decoder,
        clip_model=clip_model,
        **params,
    ).to(DEVICE)

    state = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )

    load_model_wo_clip(model, state)
    model.eval()

    return model


def load_motion():
    motion_path = pjoin(
        DATAPATH,
        "new_joint_vecs",
        MOTION_NAME + ".npy",
    )

    motion = np.load(motion_path).astype(np.float32)

    mean = np.load(pjoin(DATAPATH, "Mean.npy"))
    std = np.load(pjoin(DATAPATH, "Std.npy"))
    std[std < 1e-6] = 1.0

    motion = (motion - mean) / std

    # (T, 263) -> (263, 1, T) -> (1, 263, 1, T)
    x = torch.from_numpy(motion).permute(1, 0).contiguous()
    x = x[:, None, :]
    x = x.unsqueeze(0).to(DEVICE)

    length = x.shape[-1]

    y = torch.zeros(1, dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([length], dtype=torch.long, device=DEVICE)

    mask = (
        torch.arange(length, device=DEVICE)[None, :]
        < lengths[:, None]
    )

    return x, y, mask, lengths


print("Loading motion...")
x, y, mask, lengths = load_motion()

print(
    f"Motion: {MOTION_NAME}, "
    f"frames={x.shape[-1]}, "
    f"shape={tuple(x.shape)}"
)

print()
print("=" * 70)
print("ABSOLUTE NOISE EXPERIMENT")
print("z = mu + beta * epsilon")
print("=" * 70)


# One fixed epsilon for BOTH models and ALL beta values.
torch.manual_seed(1234)


with torch.no_grad():

    reference_eps = None

    for spec in MODELS:

        print()
        print(f"MODEL: {spec['name']}")
        print("-" * 70)

        model = load_one_model(
            spec["opt"],
            spec["ckpt"],
        )

        encoded = model.encoder(
            {
                "x": x,
                "y": y,
                "mask": mask,
            }
        )

        mu = encoded["mu"]
        logvar = encoded["logvar"]
        posterior_std = torch.exp(0.5 * logvar)

        print(
            f"mu std       = {mu.std().item():.6f}"
        )
        print(
            f"posterior std = {posterior_std.mean().item():.6f}"
        )

        # Same epsilon for both models.
        if reference_eps is None:
            reference_eps = torch.randn_like(mu)

        eps = reference_eps

        for beta in BETAS:

            z = mu + beta * eps

            decoded = model.decoder(
                {
                    "z": z,
                    "y": y,
                    "mask": mask,
                    "lengths": lengths,
                }
            )

            out = decoded["output"]

            mse = torch.mean(
                (out - x) ** 2
            ).item()

            print(
                f"beta={beta:>4}: "
                f"reconstruction_mse={mse:.6f}"
            )

        del model
        torch.cuda.empty_cache()
