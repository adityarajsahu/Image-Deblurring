import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import cv2
import os
import tensorflow as tf
from utils import process_images

focused_frames_path = 'Dataset/sharp'
defocused_frames_path = 'Dataset/defocused_blurred'

focused_frames = process_images(focused_frames_path)
defocused_frames = process_images(defocused_frames_path)

print(len(focused_frames), len(defocused_frames))
