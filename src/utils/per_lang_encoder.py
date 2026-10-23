import torch
import torch.nn as nn


# One monolingual encoder per language, each with its own trainable
# projection into the shared 512-dim space the motion decoder expects.
#
# These encoders have never seen each other. CamemBERT's "une personne
# marche" and BETO's "una persona camina" start in unrelated geometries,
# and only the projections are trained to bring them together, from ~22k
# clips. M-CLIP needed 3.4M caption pairs for the same job, so expect
# text_cosine to plateau higher than the 0.127 M-CLIP reached. That gap is
# the result, not a bug.
DEFAULT_ENCODERS = {
    'en': 'bert-base-uncased',
    'fr': 'camembert-base',
    'es': 'dccuchile/bert-base-spanish-wwm-cased',
    'pt': 'neuralmind/bert-base-portuguese-cased',
    'ru': 'DeepPavlov/rubert-base-cased',
    'zh': 'bert-base-chinese',
    'ar': 'aubmindlab/bert-base-arabertv2',
}


class PerLanguageTextEncoder(nn.Module):
    """Separate frozen encoder per language, trainable projection each.

        fr -> camembert -> 768 -> proj_fr -> 512  \\
        es -> BETO      -> 768 -> proj_es -> 512   >-- shared space -> decoder
        pt -> BERTimbau -> 768 -> proj_pt -> 512  /

    Captions arrive tagged as "[fr] une personne marche" so the right
    encoder can be selected; the tag is stripped before tokenising.

    encoders_device lets the backbones sit on CPU while the projections
    stay on GPU. Four BERT-base in fp32 is roughly 1.7GB before
    activations, so that matters when the card is shared.
    """

    def __init__(self, langs, out_dim=512, device='cuda:0',
                 encoders_device=None, names=None, max_length=128):
        super().__init__()
        import transformers

        self.langs = list(langs)
        self.out_dim = out_dim
        self._device = device
        self._enc_device = encoders_device or device
        self.max_length = max_length
        names = names or DEFAULT_ENCODERS

        self.tokenizers = {}
        self.encoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()

        prev = (torch.get_default_device()
                if hasattr(torch, 'get_default_device') else None)
        torch.set_default_device('cpu')
        try:
            for lg in self.langs:
                if lg not in names:
                    raise KeyError(f'no encoder configured for language "{lg}"')
                name = names[lg]
                print(f'  [{lg}] loading {name} ...')
                tok = transformers.AutoTokenizer.from_pretrained(name)
                enc = transformers.AutoModel.from_pretrained(name)
                hidden = enc.config.hidden_size

                self.tokenizers[lg] = tok
                self.encoders[lg] = enc
                self.projections[lg] = nn.Linear(hidden, out_dim)
        finally:
            if prev is not None:
                torch.set_default_device(prev)

        # backbones frozen, projections trained
        self.encoders = self.encoders.to(self._enc_device).float().eval()
        for p in self.encoders.parameters():
            p.requires_grad = False
        self.projections = self.projections.to(device).float()
        for p in self.projections.parameters():
            p.requires_grad = True

        n_enc = sum(p.numel() for p in self.encoders.parameters())
        n_prj = sum(p.numel() for p in self.projections.parameters())
        print(f'  per-language encoders: {len(self.langs)} backbones '
              f'({n_enc/1e6:.0f}M frozen, on {self._enc_device}), '
              f'{n_prj/1e6:.2f}M trainable projections')

    # ---- language tag handling ----
    @staticmethod
    def split_lang(t):
        """'[fr] une personne marche' -> ('fr', 'une personne marche')"""
        t = str(t)
        if t.startswith('[') and '] ' in t[:8]:
            i = t.index('] ')
            return t[1:i], t[i + 2:]
        return None, t

    # ---- forward ----
    def encode_text_raw(self, texts):
        """(B,) strings -> (B, out_dim), grouped by language so each
        backbone runs once per batch rather than once per sample."""
        pairs = [self.split_lang(t) for t in texts]
        out = torch.zeros(len(texts), self.out_dim, device=self._device)

        groups = {}
        for i, (lg, txt) in enumerate(pairs):
            if lg is None or lg not in self.encoders:
                lg = self.langs[0]          # untagged: fall back to the first
            groups.setdefault(lg, ([], []))
            groups[lg][0].append(i)
            groups[lg][1].append(txt)

        for lg, (idxs, chunk) in groups.items():
            tok = self.tokenizers[lg](
                chunk, padding=True, truncation=True,
                max_length=self.max_length,
                return_tensors='pt').to(self._enc_device)
            with torch.no_grad():
                h = self.encoders[lg](**tok)[0]          # (n, T, hidden)
                att = tok['attention_mask']
                # mean pool over real tokens, same as M-CLIP does
                pooled = (h * att.unsqueeze(2)).sum(1) / att.sum(1)[:, None]
            pooled = pooled.to(self._device).float()
            out[idxs] = self.projections[lg](pooled)

        return out

    # ---- compatibility with the CLIP interface MotionCLIP expects ----
    @property
    def ln_final(self):
        """MotionCLIP reads clip_model.ln_final.normalized_shape[0] to learn
        the text embedding width."""
        class _Shape:
            normalized_shape = (self.out_dim,)
        return _Shape()

    def encode_text(self, tokens):
        raise RuntimeError('this encoder takes raw strings; use encode_text_raw()')

    def encode_image(self, images):
        raise RuntimeError("no image tower here -- run with --clip_image_losses ''")
