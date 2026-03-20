import os
import sys

_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
if _ROOT_DIR not in sys.path:
	sys.path.insert(0, _ROOT_DIR)

from utils import train_labels_main  # noqa: F401

