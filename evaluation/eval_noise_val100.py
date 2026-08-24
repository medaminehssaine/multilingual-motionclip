import yaml
import torch
import numpy as np
from os.path import join as pjoin

from src.utils.mclip_wrapper import MClipTextEncoder
from src.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER
from src.models.modeltype.motionclip import MOTIONCLIP
from src.utils.misc import load_model_wo_clip


DEVICE = torch.device("cuda:0")
DATAPATH = "./data/HumanML3D"

MODELS = [
    (
        "KL1e-3_original",
        "./exps/vae-en-mu-kl1e-3/checkpoint_0020.pth.tar",
        "./exps/vae-en-mu-kl1e-3/opt.yaml",
    ),
    (
        "KL1e-3_beta010_ft",
        "./exps/vae-en-mu-beta10-ft/checkpoint_0025.pth.tar",
        "./exps/vae-en-mu-kl1e-3/opt.yaml",
    ),
]

BETAS = [0.0, 0.1, 0.25, 0.5, 1.0]
N_MOTIONS = 100
SEED = 1234

torch.manual_seed(SEED)
np.random.seed(SEED)


def build_model(opt_path, ckpt_path):
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


def get_valid_motion_names():
    split_path = pjoin(DATAPATH, "val.txt")
    motion_dir = pjoin(DATAPATH, "new_joint_vecs")

    with open(split_path, "r") as f:
        names = [x.strip() for x in f if x.strip()]

    valid = []

    for name in names:
        path = pjoin(motion_dir, name + ".npy")

        if not __import__("os").path.exists(path):
            continue

        try:
            motion = np.load(path, mmap_mode="r")
        except Exception:
            continue

        if 40 <= len(motion) < 200:
            valid.append(name)

        if len(valid) >= N_MOTIONS:
            break

    if len(valid) < N_MOTIONS:
        raise RuntimeError(
            f"Only found {len(valid)} valid motions, expected {N_MOTIONS}"
        )

    return valid


def prepare_motion(name):
    motion_path = pjoin(
        DATAPATH,
        "new_joint_vecs",
        name + ".npy",
    )

    motion = np.load(motion_path).astype(np.float32)

    mean = np.load(pjoin(DATAPATH, "Mean.npy"))
    std = np.load(pjoin(DATAPATH, "Std.npy"))
    std[std < 1e-6] = 1.0

    motion = (motion - mean) / std

    x = torch.from_numpy(motion).permute(1, 0).contiguous()
    x = x[:, None, :]
    x = x.unsqueeze(0).to(DEVICE)

    length = x.shape[-1]

    y = torch.zeros(
        1,
        dtype=torch.long,
        device=DEVICE,
    )

    lengths = torch.tensor(
        [length],
        dtype=torch.long,
        device=DEVICE,
    )

    mask = (
        torch.arange(length, device=DEVICE)[None, :]
        < lengths[:, None]
    )

    return x, y, mask, lengths


print("Finding motions...")
motion_names = get_valid_motion_names()
print(f"Using {len(motion_names)} motions.")

results = {
    model_name: {beta: [] for beta in BETAS}
    for model_name, _, _ in MODELS
}

for model_name, ckpt, opt in MODELS:

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    model = build_model(opt, ckpt)

    for i, name in enumerate(motion_names):

        x, y, mask, lengths = prepare_motion(name)

        with torch.no_grad():

            encoded = model.encoder({
                "x": x,
                "y": y,
                "mask": mask,
            })

            mu = encoded["mu"]

            # One fixed epsilon for THIS motion.
            # Both models see the same epsilon.
            eps = torch.randn_like(mu)

            for beta in BETAS:

                z = mu + beta * eps

                decoded = model.decoder({
                    "z": z,
                    "y": y,
                    "mask": mask,
                    "lengths": lengths,
                })

                out = decoded["output"]

                mse = torch.mean(
                    (out - x) ** 2
                ).item()

                results[model_name][beta].append(mse)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(motion_names)} motions")

    del model
    torch.cuda.empty_cache()


print()
print("=" * 70)
print("FINAL RESULTS: mean ± std over 100 motions")
print("=" * 70)

for model_name in results:

    print()
    print(model_name)
    print("-" * 70)

    for beta in BETAS:

        values = np.asarray(results[model_name][beta])

        print(
            f"beta={beta:>4}: "
            f"{values.mean():.6f} ± {values.std():.6f}"
        )
