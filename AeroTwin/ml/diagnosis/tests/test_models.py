"""
Unit tests for Diagnostic Models (Baseline, Random Forest, HistGradientBoosting, PyTorch MLP, Scorer, Evaluator).
"""

import unittest
import numpy as np
import pandas as pd
from AeroTwin.ml.diagnosis.baselines import RuleBaselineClassifier
from AeroTwin.ml.diagnosis.random_forest import RandomForestDiagnosisModel
from AeroTwin.ml.diagnosis.gradient_boosting import GradientBoostingDiagnosisModel
from AeroTwin.ml.diagnosis.neural_network import PyTorchFaultClassifier
from AeroTwin.ml.diagnosis.scoring import DiagnosisScorer, FaultDiagnosis
from AeroTwin.ml.diagnosis.evaluation import Evaluator


class TestModels(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.feature_names = ["feat1", "feat2", "res_res_signed_cylinder_3_torque_mean"]
        X_data = np.random.randn(n, len(self.feature_names))
        # Ensure distinct feature cluster for CYLINDER_3
        y_data = np.random.choice([0, 1, 2, 3, 4, 5], size=n)

        self.df_X = pd.DataFrame(X_data, columns=self.feature_names)
        self.y = y_data

    def test_rule_baseline(self):
        baseline = RuleBaselineClassifier()
        probs = baseline.predict_proba(self.df_X)
        preds = baseline.predict(self.df_X)
        self.assertEqual(probs.shape, (len(self.df_X), 6))
        self.assertEqual(len(preds), len(self.df_X))

    def test_random_forest(self):
        rf = RandomForestDiagnosisModel(n_estimators=10, max_depth=4, random_state=42)
        rf.fit(self.df_X, self.y)
        probs = rf.predict_proba(self.df_X)
        preds = rf.predict(self.df_X)
        self.assertEqual(probs.shape, (len(self.df_X), 6))
        self.assertEqual(len(preds), len(self.df_X))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_gradient_boosting(self):
        gb = GradientBoostingDiagnosisModel(max_iter=10, random_state=42)
        gb.fit(self.df_X, self.y)
        probs = gb.predict_proba(self.df_X)
        preds = gb.predict(self.df_X)
        self.assertEqual(probs.shape, (len(self.df_X), 6))
        self.assertEqual(len(preds), len(self.df_X))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_pytorch_mlp(self):
        mlp = PyTorchFaultClassifier(hidden_dim1=16, hidden_dim2=8, epochs=5, random_seed=42)
        mlp.fit(self.df_X, self.y)
        probs = mlp.predict_proba(self.df_X)
        preds = mlp.predict(self.df_X)
        self.assertEqual(probs.shape, (len(self.df_X), 6))
        self.assertEqual(len(preds), len(self.df_X))
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_diagnosis_scorer(self):
        rf = RandomForestDiagnosisModel(n_estimators=10, random_state=42).fit(self.df_X, self.y)
        scorer = DiagnosisScorer(model=rf, model_name="random_forest", feature_configuration="HYBRID")
        win_X = self.df_X.iloc[0:1]
        record = scorer.diagnose_window(win_X, timestamp=1.5, sim_time=1.5)

        self.assertIsInstance(record, FaultDiagnosis)
        self.assertEqual(record.model_name, "random_forest")
        self.assertEqual(len(record.probabilities), 6)
        self.assertAlmostEqual(sum(record.probabilities.values()), 1.0, places=4)

    def test_evaluator(self):
        evaluator = Evaluator()
        metrics = evaluator.evaluate_predictions(self.y, self.y, experiment_name="Test")
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(len(metrics["confusion_matrix"]), 6)


if __name__ == "__main__":
    unittest.main()
