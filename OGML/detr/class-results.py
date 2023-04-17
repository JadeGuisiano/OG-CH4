import torch
import pandas as pd
from pathlib import Path

def load_eval(eval_path):
    data = torch.load(eval_path)
    # precision is n_iou, n_points, n_cat, n_area, max_det
    precision = data['precision']
    # take precision for all classes, all areas and 100 detections
    CLASSES = [
        'N/A', 'well','tank','compressor'
    ]
    CLASSES = [c for c in CLASSES if c != 'N/A']
    area = 0
    return pd.DataFrame.from_dict({c: p for c, p in zip(CLASSES, precision[0, :, :, area, -1].mean(0) * 100)}, orient='index')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    df = load_eval('./output/eval/latest.pth')
    display(df)
