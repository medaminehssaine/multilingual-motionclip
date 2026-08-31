import json, numpy as np, torch, sys
sys.argv = ['x']
import importlib.util
spec = importlib.util.spec_from_file_location("ela", "eval_latent_alignment.py")
ela = importlib.util.module_from_spec(spec); spec.loader.exec_module(ela)
