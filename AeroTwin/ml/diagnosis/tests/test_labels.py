"""
Unit tests for Fault Labels and Encoding Utilities.
"""

import unittest
import numpy as np
from AeroTwin.ml.diagnosis.labels import (
    FaultClass,
    encode_fault_label,
    decode_fault_label,
    normalize_target_string,
    compute_class_weights,
)


class TestLabels(unittest.TestCase):

    def test_normalize_target_string(self):
        self.assertEqual(normalize_target_string("CYLINDER", "CYLINDER_1"), "CYLINDER_1")
        self.assertEqual(normalize_target_string("CYLINDER", "CYL1"), "CYLINDER_1")
        self.assertEqual(normalize_target_string("CYLINDER", "CYLINDER_3"), "CYLINDER_3")
        self.assertEqual(normalize_target_string("BEARING", "BEARING"), "BEARING")
        self.assertEqual(normalize_target_string("COOLING", "COOLING"), "COOLING")
        self.assertEqual(normalize_target_string("LUBRICATION", "OIL_PUMP"), "LUBRICATION")
        self.assertEqual(normalize_target_string("HEALTHY", "NONE"), "HEALTHY")

    def test_encode_and_decode_fault_label(self):
        self.assertEqual(encode_fault_label("CYLINDER_3"), 2)
        self.assertEqual(decode_fault_label(2), "CYLINDER_3")
        self.assertEqual(encode_fault_label("HEALTHY"), 0)
        self.assertEqual(decode_fault_label(0), "HEALTHY")

    def test_compute_class_weights(self):
        y = np.array([0, 0, 0, 1, 2, 3, 4, 5])
        weights = compute_class_weights(y)
        self.assertEqual(len(weights), 6)
        self.assertTrue(np.all(weights > 0.0))


if __name__ == "__main__":
    unittest.main()
