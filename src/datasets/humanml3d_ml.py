import os
import json
import random
import codecs as cs
import numpy as np
import torch
from os.path import join as pjoin
from tqdm import tqdm

from .dataset import Dataset


class HumanML3DML(Dataset):
    """HumanML3D with multilingual captions.

    Reads captions from captions_<lang>.json rather than texts/*.txt, so that
    the English baseline and the multilingual model draw from exactly the same
    caption pool (the FIRST caption of each clip). Without this the English
    model would see several captions per clip and the multilingual one only
    one, which would confound the comparison.

    --langs en                 English only, the controlled baseline
    --langs en fr es pt ru zh ar   one language sampled per clip per epoch

    Sampling is per-__getitem__, so over 100 epochs each clip is seen in every
    language roughly equally. Dataset size is unchanged, so epoch time and the
    number of gradient steps match the baseline exactly.
    """

    dataname = "humanml3d_ml"

    SPLIT_FILES = {"train": "train.txt", "val": "val.txt",
                   "vald": "val.txt", "test": "test.txt"}

    def __init__(self, datapath="./data/HumanML3D", split="train",
                 langs=("en",), caption_dir=".",
                 min_motion_len=40, max_motion_len=200,
                 normalize=True, clip_preprocess=None, **kargs):

        self.datapath = datapath
        self.langs = list(langs)
        self.normalize = normalize
        self.min_motion_len = min_motion_len
        self.max_motion_len = max_motion_len
        self.clip_preprocess = clip_preprocess

        base_split = "val" if split == "vald" else split
        super().__init__(split=base_split, **kargs)
        self.split = base_split

        motion_dir = pjoin(datapath, "new_joint_vecs")

        self.mean = np.load(pjoin(datapath, "Mean.npy"))
        self.std = np.load(pjoin(datapath, "Std.npy"))
        self.std[self.std < 1e-6] = 1.0

        # ---- captions, one file per language ----
        caps = {}
        for lg in self.langs:
            p = pjoin(caption_dir, f"captions_{lg}.json")
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"missing {p} -- run translate_nllb.py --lang {lg}")
            caps[lg] = json.load(open(p))
        n_ref = len(caps[self.langs[0]])
        for lg in self.langs[1:]:
            if len(caps[lg]) != n_ref:
                print(f"  WARNING: captions_{lg}.json has {len(caps[lg])} "
                      f"entries, captions_{self.langs[0]}.json has {n_ref}")

        split_file = pjoin(datapath, self.SPLIT_FILES[split])
        with cs.open(split_file, "r") as f:
            id_list = [l.strip() for l in f if l.strip()]

        self._motions, self._caps, self._names = [], [], []
        n_short = n_long = n_missing = n_notext = 0

        for name in tqdm(id_list, desc=f"humanml3d_ml[{split}]"):
            mpath = pjoin(motion_dir, name + ".npy")
            if not os.path.exists(mpath):
                n_missing += 1
                continue

            motion = np.load(mpath)
            if len(motion) < self.min_motion_len:
                n_short += 1
                continue
            if len(motion) >= self.max_motion_len:
                n_long += 1
                continue

            # need the caption in every requested language
            per_lang = {}
            for lg in self.langs:
                c = caps[lg].get(name)
                if c:
                    per_lang[lg] = c
            if len(per_lang) != len(self.langs):
                n_notext += 1
                continue

            self._motions.append(motion.astype(np.float32))
            self._caps.append(per_lang)
            self._names.append(name)

        n = len(self._motions)
        if n == 0:
            raise RuntimeError(f"no clips loaded for split '{split}'")

        self._num_frames_in_video = np.array([len(m) for m in self._motions])

        self._actions = [0] * n
        self._action_classes = ["dummy"]
        self.num_classes = 1
        self._action_to_label = {0: 0}
        self._label_to_action = {0: 0}

        self.clip_label_text = "text_raw_labels"
        self.db = {self.clip_label_text: None}

        idx = list(range(n))
        self._train = idx
        self._test = idx

        print(f"humanml3d_ml[{split}]: {n} clips, langs={','.join(self.langs)}  "
              f"(skipped {n_short} short, {n_long} long, "
              f"{n_missing} missing, {n_notext} untranslated)")

    # ---- motion ----
    def _load(self, ind, frame_ix):
        motion = self._motions[ind][frame_ix]
        if self.normalize:
            motion = (motion - self.mean) / self.std
        ret = torch.from_numpy(motion.astype(np.float32))
        return ret.permute(1, 0)[:, None, :].contiguous()      # (263, 1, T)

    # ---- text: one language sampled per access ----
    def get_clip_text(self, ind, frame_ix):
        lg = random.choice(self.langs)
        cap = self._caps[ind][lg]
        return np.array([cap] * len(frame_ix), dtype=object)

    # ---- helpers ----
    def inv_transform(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(self.std, device=x.device, dtype=x.dtype) \
                     + torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        return x * self.std + self.mean

    def get_name(self, ind):
        return self._names[ind]

    def caption(self, ind, lang="en"):
        return self._caps[ind].get(lang)
