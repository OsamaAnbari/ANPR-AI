import os
import sys
from PIL import Image
from IPython.display import display
import cv2
import numpy as np # 1.20
import pathlib
import csv
import datetime

#from collections import defaultdict
#import mysql.connector
#import tarfile
#import zipfile
#from io import StringIO
#from matplotlib import pyplot as plt

import tensorflow as tf # pip install --ignore-installed --upgrade tensorflow==2.5.0
from tensorflow import keras

from object_detection.utils import ops as utils_ops  # python -m pip install .      protpbuf 3.2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util