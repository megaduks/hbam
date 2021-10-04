# HBAM: Hierarchical Bitmap Adjacency Matrix

## Description

The goal of the project is to verify the usefulness of the hierarchical bitmap representation of adjacency matrices for graph comparison, compression, and similarity search.

### Notebook versioning

In order to version-control Jupyter notebooks, please install [JupyText](https://jupytext.readthedocs.io/en/latest/index.html)

```
pip install jupytext --upgrade 
```

Jupyter notebook source files `*.ipynb` are added to `.gitignore`, if you want to upload a new version of a notebook, first convert it to regular Python file.

```python
jupytext --to py <filename>.ipynb
```

To convert back from `*.py` to notebook format simply run

```python
jupytext --to ipynb <filename>.py
```


