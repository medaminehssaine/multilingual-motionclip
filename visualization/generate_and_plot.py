import os
import sys
sys.path.append('.')
import numpy as np
import torch
import clip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils.get_model_and_data import get_model_and_data
from src.parser.visualize import parser
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa

CHAIN = [
    [8, 3, 6, 9, 12],
    [9, 13, 16],
    [9, 14, 15, 17],
    [8, 1, 4, 7, 10],
    [8, 2, 5, 0, 11],
]
COLORS = ['purple', 'red', 'blue', 'green', 'orange']


def plot_motion_gif(joints, title, outpath, fps=20):
    allpts = joints.reshape(-1, 3)
    max_range = (allpts.max(0) - allpts.min(0)).max() / 2.0
    mid = (allpts.max(0) + allpts.min(0)) / 2.0
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')

    def update(index):
        ax.clear()
        pose = joints[index]
        for chain, color in zip(CHAIN, COLORS):
            cp = pose[chain]
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], c=color, linewidth=3)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        ax.view_init(elev=10, azim=90)
        ax.set_title(title, fontsize=10)

    ani = FuncAnimation(fig, update, frames=int(joints.shape[0]), interval=50, repeat=True)
    ani.save(outpath, writer='pillow', fps=fps)
    plt.close(fig)


def main():
    parameters, folder, checkpointname, epoch = parser()
    parameters["device"] = "cuda:0"
    model, datasets = get_model_and_data(parameters, split='vald')
    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)

    assert os.path.isfile(parameters['input_file'])
    with open(parameters['input_file'], 'r') as fr:
        texts = [s.replace('\n', '') for s in fr.readlines()]
    texts = [t for t in texts if t.strip() != '']
    print(f"Loaded {len(texts)} text prompts")

    clip_tokens = clip.tokenize(texts).to(parameters['device'])
    clip_features = model.clip_model.encode_text(clip_tokens).float().unsqueeze(0)
    gendurations = torch.ones((len(texts), 1), dtype=int) * parameters['num_frames']

    model.eval()
    with torch.no_grad():
        generation = model.generate(clip_features, gendurations,
                                    is_amass=True, is_clip_features=True)

    xyz = generation['output_xyz'].cpu().numpy()
    print("output_xyz shape:", xyz.shape)

    outdir = os.path.join(folder, "generated_skeletons")
    os.makedirs(outdir, exist_ok=True)

    for i, text in enumerate(texts):
        motion = xyz[i]
        motion = np.transpose(motion, (2, 0, 1))
        safe = "".join(c if c.isalnum() else "_" for c in text)[:40]
        outpath = os.path.join(outdir, f"gen_{i:02d}_{safe}.gif")
        plot_motion_gif(motion, text, outpath)
        print(f"Saved {outpath}")

    print(f"\nDone. {len(texts)} generated motions saved to {outdir}")


if __name__ == '__main__':
    main()
