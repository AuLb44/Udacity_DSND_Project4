"""
Basic smoke tests for the Recommendations with IBM project.

These tests verify that core dependencies are installed and importable,
ensuring the basic environment setup is correct.
"""

import pytest


def test_numpy_import():
    """Test that numpy can be imported successfully."""
    import numpy as np
    assert np is not None
    assert hasattr(np, '__version__')


def test_pandas_import():
    """Test that pandas can be imported successfully."""
    import pandas as pd
    assert pd is not None
    assert hasattr(pd, '__version__')


def test_matplotlib_import():
    """Test that matplotlib can be imported successfully."""
    import matplotlib.pyplot as plt
    assert plt is not None


def test_sklearn_import():
    """Test that scikit-learn can be imported successfully."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    assert TruncatedSVD is not None
    assert cosine_similarity is not None
    assert TfidfVectorizer is not None
    assert KMeans is not None


def test_scipy_import():
    """Test that scipy can be imported successfully."""
    import scipy
    from scipy.sparse.linalg import svds
    
    assert scipy is not None
    assert svds is not None


def test_notebook_file_exists():
    """Test that the main notebook file exists."""
    import os
    from pathlib import Path
    
    # Get the repository root (parent of tests directory)
    repo_root = Path(__file__).parent.parent
    notebook_path = repo_root / 'Recommendations_with_IBM.ipynb'
    
    assert notebook_path.exists(), f"Notebook file not found: {notebook_path}"


def test_numpy_basic_operation():
    """Test basic numpy operations work correctly."""
    import numpy as np
    arr = np.array([1, 2, 3, 4, 5])
    assert arr.mean() == 3.0
    assert arr.sum() == 15
    assert len(arr) == 5


def test_pandas_basic_operation():
    """Test basic pandas operations work correctly."""
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    assert len(df) == 3
    assert list(df.columns) == ['a', 'b']
    assert df['a'].sum() == 6


def test_sklearn_basic_operation():
    """Test basic sklearn operations work correctly."""
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    
    # Create a simple matrix
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    # Apply SVD
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_transformed = svd.fit_transform(X)
    
    # Check the shape of the result
    assert X_transformed.shape == (3, 2)
