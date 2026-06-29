"""
Testy walidacyjne dla Projektu 6: NCG jako solver regresji logistycznej.

Uruchomienie z katalogu projektu:
    python test_validation.py
albo, jeśli plik znajduje się w tests/:
    python tests/test_validation.py

Testy wczytują notebook projektu, wykonują tylko komórki potrzebne do zdefiniowania danych
oraz funkcji, a następnie sprawdzają poprawność funkcji straty, gradientu, hesjanu i solverów.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _find_notebook() -> Path:
    """Znajdź notebook projektu w typowych lokalizacjach."""
    candidates = [
        Path("Projekt_6.ipynb"),
        Path("notebooks/Projekt_6.ipynb"),
        Path("/mnt/data/Projekt_6.ipynb"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Nie znaleziono notebooka projektu. Oczekiwano np. Projekt_6.ipynb "
    )


def load_project_namespace() -> dict:
    """
    Wykonaj z notebooka tylko komórki potrzebne do testów.
    Pomijamy komórki z długimi eksperymentami, tabelami i wykresami.
    """
    notebook_path = _find_notebook()
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__validation__"}

    required_markers = [
        "import numpy as np",
        "data = load_breast_cancer()",
        "y = 2 * y - 1",
        "X_train, X_test, y_train, y_test = train_test_split",
        "scaler = StandardScaler()",
        "X_train = np.c_[np.ones",
        "def logistic_loss",
        "def logistic_gradient",
        "def logistic_hessian",
        "def predict_labels",
        "def gradient_descent",
        "def armijo_backtracking",
        "def ncg_prp_plus",
        "def newton_raphson",
    ]

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if any(marker in source for marker in required_markers):
            exec(compile(source, str(notebook_path), "exec"), namespace)

    expected = [
        "X_train", "X_test", "y_train", "y_test",
        "logistic_loss", "logistic_gradient", "logistic_hessian",
        "gradient_descent", "ncg_prp_plus", "newton_raphson", "accuracy",
    ]
    missing = [name for name in expected if name not in namespace]
    assert not missing, f"Brak wymaganych nazw po wykonaniu notebooka: {missing}"
    return namespace


def test_data_preparation(ns: dict) -> None:
    X_train, X_test = ns["X_train"], ns["X_test"]
    y_train, y_test = ns["y_train"], ns["y_test"]
    assert X_train.shape == (455, 31), f"Nieoczekiwany rozmiar X_train: {X_train.shape}"
    assert X_test.shape == (114, 31), f"Nieoczekiwany rozmiar X_test: {X_test.shape}"
    assert set(np.unique(y_train)) == {-1, 1}
    assert set(np.unique(y_test)) == {-1, 1}
    assert np.allclose(X_train[:, 0], 1.0), "Pierwsza kolumna powinna być biasem równym 1."


def test_loss_gradient_hessian(ns: dict) -> None:
    X_train, y_train = ns["X_train"], ns["y_train"]
    loss = ns["logistic_loss"]
    grad = ns["logistic_gradient"]
    hess = ns["logistic_hessian"]

    w0 = np.zeros(X_train.shape[1])
    loss0 = loss(w0, X_train, y_train)
    g0 = grad(w0, X_train, y_train)
    H0 = hess(w0, X_train, y_train)

    assert math.isfinite(loss0)
    assert abs(loss0 - math.log(2.0)) < 1e-12, "Dla w=0 strata powinna wynosić log(2)."
    assert g0.shape == w0.shape
    assert H0.shape == (X_train.shape[1], X_train.shape[1])
    assert np.allclose(H0, H0.T, atol=1e-10), "Hesjan powinien być symetryczny."
    assert np.linalg.eigvalsh(H0).min() > -1e-8, "Hesjan straty logistycznej powinien być PSD."


def test_gradient_by_finite_differences(ns: dict) -> None:
    X_train, y_train = ns["X_train"], ns["y_train"]
    loss = ns["logistic_loss"]
    grad = ns["logistic_gradient"]

    rng = np.random.default_rng(123)
    w = rng.normal(scale=0.1, size=X_train.shape[1])
    g = grad(w, X_train, y_train)
    eps = 1e-6

    for j in [0, 1, 2, 10, X_train.shape[1] - 1]:
        e = np.zeros_like(w)
        e[j] = 1.0
        numerical = (loss(w + eps * e, X_train, y_train) - loss(w - eps * e, X_train, y_train)) / (2 * eps)
        assert abs(numerical - g[j]) < 1e-5, (
            f"Gradient nie zgadza się z różnicami skończonymi dla j={j}: "
            f"num={numerical}, analityczny={g[j]}"
        )


def test_descent_and_armijo(ns: dict) -> None:
    X_train, y_train = ns["X_train"], ns["y_train"]
    loss = ns["logistic_loss"]
    grad = ns["logistic_gradient"]
    armijo = ns["armijo_backtracking"]

    w = np.zeros(X_train.shape[1])
    g = grad(w, X_train, y_train)
    d = -g
    alpha = armijo(w, d, X_train, y_train)
    assert alpha > 0
    assert loss(w + alpha * d, X_train, y_train) < loss(w, X_train, y_train)


def test_solvers_reduce_loss_and_predict_well(ns: dict) -> None:
    X_train, X_test = ns["X_train"], ns["X_test"]
    y_train, y_test = ns["y_train"], ns["y_test"]
    loss = ns["logistic_loss"]
    accuracy = ns["accuracy"]
    w0 = np.zeros(X_train.shape[1])
    start_loss = loss(w0, X_train, y_train)

    gd = ns["gradient_descent"](X_train, y_train, alpha=0.1, max_iter=200, tol=1e-8)
    ncg = ns["ncg_prp_plus"](X_train, y_train, max_iter=200, tol=1e-8)
    newton = ns["newton_raphson"](X_train, y_train, max_iter=20, tol=1e-8)

    for name, result in [("GD", gd), ("NCG-PRP+", ncg), ("Newton", newton)]:
        assert result["final_loss"] < start_loss, f"{name}: funkcja straty nie spadła."
        assert len(result["loss_hist"]) == result["iterations"]
        assert np.all(np.isfinite(result["loss_hist"])), f"{name}: historia straty zawiera NaN/inf."
        test_acc = accuracy(result["w"], X_test, y_test)
        assert test_acc >= 0.90, f"{name}: accuracy testowe jest za niskie: {test_acc}"


def main() -> None:
    ns = load_project_namespace()
    tests = [
        test_data_preparation,
        test_loss_gradient_hessian,
        test_gradient_by_finite_differences,
        test_descent_and_armijo,
        test_solvers_reduce_loss_and_predict_well,
    ]
    for test in tests:
        test(ns)
        print(f"OK: {test.__name__}")
    print("\nWszystkie testy walidacyjne dla projektu 6 zakończone powodzeniem.")


if __name__ == "__main__":
    main()
