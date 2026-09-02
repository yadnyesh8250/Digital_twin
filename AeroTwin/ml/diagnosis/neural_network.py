"""
AeroTwin-4 Model 4: PyTorch Supervised MLP Neural Fault Classifier.

3-layer feed-forward neural network trained via class-weighted CrossEntropyLoss.
"""

from typing import Optional, Dict, Any, Tuple, List
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from AeroTwin.ml.diagnosis.labels import FAULT_CLASS_ORDER, compute_class_weights


class DiagnosisMLPNet(nn.Module):
    """
    Feed-Forward Supervised Classifier Neural Network.
    """

    def __init__(self, input_dim: int, hidden_dim1: int = 64, hidden_dim2: int = 32, num_classes: int = 6, dropout_rate: float = 0.2):
        super(DiagnosisMLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchFaultClassifier:
    """
    PyTorch Supervised Multi-Class Fault Diagnosis Classifier.
    """

    def __init__(
        self,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        lr: float = 1e-3,
        epochs: int = 60,
        batch_size: int = 32,
        dropout_rate: float = 0.2,
        random_seed: int = 42,
    ):
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        self.net: Optional[DiagnosisMLPNet] = None
        self.input_dim: int = 0
        self.num_classes: int = len(FAULT_CLASS_ORDER)
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "PyTorchFaultClassifier":
        """
        Fit PyTorch MLP on training features X and class targets y.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_mat = X_train.values
        else:
            X_mat = np.array(X_train)

        y_indices = np.array(y_train, dtype=np.int64)
        self.input_dim = X_mat.shape[1]

        class_weights = compute_class_weights(y_indices)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

        self.net = DiagnosisMLPNet(
            input_dim=self.input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
        )

        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

        tensor_x = torch.tensor(X_mat, dtype=torch.float32)
        tensor_y = torch.tensor(y_indices, dtype=torch.long)

        best_loss = float("inf")
        patience = 12
        patience_counter = 0

        self.net.train()
        for epoch in range(self.epochs):
            permutation = torch.randperm(tensor_x.size(0))

            for i in range(0, tensor_x.size(0), self.batch_size):
                indices = permutation[i : i + self.batch_size]
                batch_x = tensor_x[indices]
                batch_y = tensor_y[indices]

                optimizer.zero_grad()
                logits = self.net(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            # Early stopping check if validation data provided
            if X_val is not None and y_val is not None:
                val_mat = X_val[self.feature_names].values if isinstance(X_val, pd.DataFrame) else np.array(X_val)
                val_x = torch.tensor(val_mat, dtype=torch.float32)
                val_y = torch.tensor(np.array(y_val, dtype=np.int64), dtype=torch.long)

                self.net.eval()
                with torch.no_grad():
                    val_logits = self.net(val_x)
                    val_loss = criterion(val_logits, val_y).item()
                self.net.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self.net.eval()
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities using softmax output.
        """
        if not self.is_fitted or self.net is None:
            raise RuntimeError("PyTorchFaultClassifier is not fitted!")

        X_mat = X[self.feature_names].values if isinstance(X, pd.DataFrame) else np.array(X)
        tensor_x = torch.tensor(X_mat, dtype=torch.float32)

        self.net.eval()
        with torch.no_grad():
            logits = self.net(tensor_x)
            probs = torch.softmax(logits, dim=1).numpy()

        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class indices (0..5).
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, model_dir: str):
        """
        Save PyTorch model weights and config JSON.
        """
        os.makedirs(model_dir, exist_ok=True)
        weights_path = os.path.join(model_dir, "mlp_weights.pt")
        config_path = os.path.join(model_dir, "mlp_config.json")

        if self.net is not None:
            torch.save(self.net.state_dict(), weights_path)
        config = {
            "input_dim": self.input_dim,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def load(self, model_dir: str) -> "PyTorchFaultClassifier":
        """
        Load PyTorch model weights and config JSON.
        """
        weights_path = os.path.join(model_dir, "mlp_weights.pt")
        config_path = os.path.join(model_dir, "mlp_config.json")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.input_dim = config["input_dim"]
        self.hidden_dim1 = config["hidden_dim1"]
        self.hidden_dim2 = config["hidden_dim2"]
        self.num_classes = config["num_classes"]
        self.dropout_rate = config["dropout_rate"]
        self.feature_names = config["feature_names"]
        self.is_fitted = config["is_fitted"]

        self.net = DiagnosisMLPNet(
            input_dim=self.input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
        )
        self.net.load_state_dict(torch.load(weights_path, weights_only=True))
        self.net.eval()
        return self
