"""
AeroTwin-4 Phase 6 Label Definitions & Mapping Utilities.

Target Fault Classes:
0: HEALTHY
1: CYLINDER_1
2: CYLINDER_3
3: BEARING
4: COOLING
5: LUBRICATION
"""

from enum import Enum
from typing import List, Dict, Union
import numpy as np


class FaultClass(str, Enum):
    HEALTHY = "HEALTHY"
    CYLINDER_1 = "CYLINDER_1"
    CYLINDER_3 = "CYLINDER_3"
    BEARING = "BEARING"
    COOLING = "COOLING"
    LUBRICATION = "LUBRICATION"


FAULT_CLASS_ORDER = [
    FaultClass.HEALTHY,
    FaultClass.CYLINDER_1,
    FaultClass.CYLINDER_3,
    FaultClass.BEARING,
    FaultClass.COOLING,
    FaultClass.LUBRICATION,
]

FAULT_TO_IDX: Dict[str, int] = {fc.value: idx for idx, fc in enumerate(FAULT_CLASS_ORDER)}
IDX_TO_FAULT: Dict[int, str] = {idx: fc.value for idx, fc in enumerate(FAULT_CLASS_ORDER)}


def normalize_target_string(gt_type: str, gt_comp: str) -> str:
    """
    Map raw ground truth fields (gt_degradation_type, gt_target_component) to canonical FaultClass string.
    """
    if str(gt_type).upper() == "HEALTHY" or str(gt_comp).upper() in ["NONE", "HEALTHY", "NAN"]:
        return FaultClass.HEALTHY.value

    gt_type_str = str(gt_type).upper()
    gt_comp_str = str(gt_comp).upper()

    if "CYLINDER_1" in gt_comp_str or "CYL1" in gt_comp_str or ("CYLINDER" in gt_type_str and "1" in gt_comp_str):
        return FaultClass.CYLINDER_1.value
    elif "CYLINDER_3" in gt_comp_str or "CYL3" in gt_comp_str or ("CYLINDER" in gt_type_str and "3" in gt_comp_str):
        return FaultClass.CYLINDER_3.value
    elif "BEARING" in gt_type_str or "BEARING" in gt_comp_str:
        return FaultClass.BEARING.value
    elif "COOLING" in gt_type_str or "COOLING" in gt_comp_str:
        return FaultClass.COOLING.value
    elif "LUBRICATION" in gt_type_str or "OIL" in gt_comp_str or "LUBRICATION" in gt_comp_str:
        return FaultClass.LUBRICATION.value

    return FaultClass.HEALTHY.value


def encode_fault_label(label_str: str) -> int:
    """
    Convert string label to integer index (0..5).
    """
    norm = normalize_target_string(label_str, label_str)
    return FAULT_TO_IDX.get(norm, 0)


def decode_fault_label(label_idx: int) -> str:
    """
    Convert integer index (0..5) to string label.
    """
    return IDX_TO_FAULT.get(int(label_idx), FaultClass.HEALTHY.value)


def compute_class_weights(y_indices: np.ndarray) -> np.ndarray:
    """
    Compute balanced class weights for training loss functions.
    weight_c = n_samples / (n_classes * count_c)
    """
    n_samples = len(y_indices)
    n_classes = len(FAULT_CLASS_ORDER)
    weights = np.ones(n_classes, dtype=np.float32)

    counts = np.bincount(y_indices, minlength=n_classes)
    for c in range(n_classes):
        if counts[c] > 0:
            weights[c] = n_samples / (n_classes * counts[c])
        else:
            weights[c] = 1.0

    return weights.astype(np.float32)
