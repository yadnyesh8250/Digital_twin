"""
AeroTwin-4 Phase 6: Supervised Fault Diagnosis & Component Identification Package.
"""

from AeroTwin.ml.diagnosis.labels import FaultClass, encode_fault_label, decode_fault_label
from AeroTwin.ml.diagnosis.scoring import FaultDiagnosis, DiagnosisScorer

__all__ = [
    "FaultClass",
    "encode_fault_label",
    "decode_fault_label",
    "FaultDiagnosis",
    "DiagnosisScorer",
]
