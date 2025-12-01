# Recommendations with IBM

[![CI](https://github.com/AuLb44/Udacity_DSND_Project4/actions/workflows/ci.yml/badge.svg)](https://github.com/AuLb44/Udacity_DSND_Project4/actions/workflows/ci.yml)

## Overview

This project analyzes user interactions with articles on the IBM Watson Studio platform to build recommendation systems. The goal is to recommend articles to users based on their previous interactions and the behavior of similar users.

The project implements multiple recommendation approaches:
- **Rank Based Recommendations**: Recommend the most popular articles
- **User-User Based Collaborative Filtering**: Recommend articles based on similar users
- **Content Based Recommendations**: Recommend articles based on content similarity
- **Matrix Factorization**: Use SVD for recommendations

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/AuLb44/Udacity_DSND_Project4.git
cd Udacity_DSND_Project4
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

### Data Requirements

This project requires data from the IBM Watson Studio platform. The notebook expects the following data files in a `data/` directory:

- `data/user-item-interactions.csv` - User interactions with articles
- `data/articles_community.csv` - Article content and metadata

**To obtain the data:**

1. Access the IBM Watson Studio platform (typically provided through the Udacity Data Science Nanodegree program)
2. Download the required datasets
3. Create a `data/` directory in the project root:
   ```bash
   mkdir data
   ```
4. Place the CSV files in the `data/` directory

**Note:** The data files are not included in this repository. You must obtain them from the IBM Watson Studio platform or through your course materials.

## Running the Project

The main analysis is contained in the Jupyter notebook:

```bash
jupyter notebook Recommendations_with_IBM.ipynb
```

This will open the notebook in your browser where you can run the cells to:
- Perform exploratory data analysis
- Build and evaluate different recommendation systems
- Compare recommendation approaches

## Running Tests

The project includes a basic test suite to verify core functionality.

Run all tests with pytest:
```bash
pytest
```

Run with verbose output:
```bash
pytest -v
```

Run specific test file:
```bash
pytest tests/test_basic.py
```

## Project Structure

```
Udacity_DSND_Project4/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── Recommendations_with_IBM.ipynb     # Main analysis notebook
├── data/                              # Data files (not included in repo)
│   ├── user-item-interactions.csv     # User interaction data
│   └── articles_community.csv         # Article content data
└── tests/                             # Test suite
    └── test_basic.py                  # Basic smoke tests
```

## Troubleshooting

### Missing Data Files

If you see errors like `FileNotFoundError: [Errno 2] No such file or directory: 'data/user-item-interactions.csv'`:
- Ensure you've created the `data/` directory in the project root
- Verify the data files are downloaded from IBM Watson Studio platform
- Check that the file names match exactly: `user-item-interactions.csv` and `articles_community.csv`

### Import Errors

If you encounter import errors, ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Jupyter Kernel Issues

If the Jupyter kernel is not found:
```bash
python -m ipykernel install --user --name=udacity-project4
```

### Test Failures

If tests fail, ensure you're using Python 3.8 or higher:
```bash
python --version
```

### Memory Issues

The recommendation algorithms can be memory-intensive. If you run into memory issues:
- Reduce the dataset size
- Use a machine with more RAM
- Close other applications

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is part of the Udacity Data Science Nanodegree program.