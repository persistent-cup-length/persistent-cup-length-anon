# 🌀 persistent-cup-length

Code accompanying the paper **"Doughnut or Mickey Mouse? Detecting Toroidal Structure in Data through Persistent Cup-Length"**.  

This repository provides an implementation and tutorial for computing the **persistent cup-length**, an algebraic-topological invariant, compatible with persistent cohomology computed using [Ripser](https://github.com/scikit-tda/ripser) (for Vietoris-Rips filtrations) and [Dionysus](https://www.mrzv.org/software/dionysus/) (for simplexwise filtrations).

---
## 🧪 Citation

If you use this code in your research, please cite the associated paper:

``` @misc{ivshina2025doughnutmickeymousedetecting,
      title={Doughnut or Mickey Mouse? Detecting Toroidal Structure in Data through Persistent Cup-Length}, 
      author={Ekaterina S. Ivshina and Galit Anikeeva and Ling Zhou},
      year={2025},
      eprint={2507.11151},
      archivePrefix={arXiv},
      primaryClass={math.AT},
      url={https://arxiv.org/abs/2507.11151}, 
}
```

## 📖 Overview

The **persistent cup-length** algorithm, introduced by [Contessoto, Mémoli, Stefanou and Zhou (2022)](https://doi.org/10.4230/lipics.socg.2022.31), identifies higher-order topological interactions in data using persistent cohomology and the cup product operation over Z mod 2. This helps detect nontrivial toroidal structures, going beyond standard barcode analysis.

This code is built for use with **Ripser** and works with both full and landmark-subsampled Vietoris–Rips filtrations. A version of the algorithm designed to work with simplexwise filtrations in [Dionysus](https://www.mrzv.org/software/dionysus/) will be released soon.

---

## 📁 Repository Contents

- `cup_length_utils.py` – Core implementation of the persistent cup-length algorithm for Vieotoris-Rips filtrations.
- `dreimac_utils.py` – Utility functions; most are from [Dreimac library](https://dreimac.scikit-tda.org/en/latest/)
- `dreimac_combinatorial.py` – Utility functions for combinatorial number system
- `tutorial_on_persistent_cup_length.ipynb` – A **Colab notebook** tutorial for running the persistent cup-length algorithm on a torus.
- `simplexwise_persistent_cup_length.ipynb` -  A **Colab notebook** containing our implementation of the persistent cup-length algorithm for simplexwise filtrations computed with [Dionysus](https://www.mrzv.org/software/dionysus/).
- `fig1_dataset` folder contains the torus distance matrix as well as the point cloud used to create the top panel (torus) of Figure 1 in our paper.

The distance matrix used to create the bottom panel (wedge sum) in Figure 1 of the paper is available [here](https://drive.google.com/drive/folders/1Gj4urjBrd7jWnAEoz1Hp7-ED4OdexG_l?usp=sharing).
 

---

## 🚀 Getting Started in Google Colab
Note that the code assumes python 3.8
### 1. Install dependencies

```bash
!pip install ripser
!pip install scipy==1.10.1
```

> ⚠️ After installing, you must **restart the runtime/session** if using Google Colab.

---

### 2. Upload files in Colab

Upload the following files to your Colab environment:

- `cup_length_utils.py`
- `dreimac_utils.py`
- `dreimac_combinatorial.py`

Then run the import cell in the notebook:

```python
from cup_length_utils import *
from dreimac_utils import *
from dreimac_combinatorial import *
```

---

### 3. Example: Computing cup length on a torus

```python
np.random.seed(42)
# Parameters
n_points = 2000  # Number of points to sample
R = 5  # Major radius of the torus
r = 2  # Minor radius of the torus
 

# Sample points from a torus
torus_points = sample_torus(n_points, R, r)

# Create a distance matrix from the point cloud
d = pairwise_distances(torus_points, metric='euclidean')
threshold = np.max(d[~np.isinf(d)])


# Compute persistent homology on a subsampled point cloud.
ripser_result = rips_filtration_diagram(data = d, distance_matrix = True, n_landmarks = 150, thresh = threshold)
dgms = ripser_result['dgms']
plot_diagrams(dgms, show=True)


# Compute the threshold persistence  
min_persistence = compute_threshold_persistence(dgms[1])


t0 = time.time()
# Compute persistent cup length
(persistent_cup_length_matrix, b_times, d_times, cup_length_2) = compute_persistent_cup_length_ripser(2, \
                                                    ripser_result, \
                                                    d, \
                                                    min_persistence, \
                                                    threshold)

t1 = time.time()
elapsed_time = (t1 - t0) / 60  

print(f"Function took {elapsed_time:.2f} minutes to run.")

nonzero_elements = persistent_cup_length_matrix[persistent_cup_length_matrix != 0]
unique_vals, counts = np.unique(nonzero_elements, return_counts=True)
for value, count in zip(unique_vals, counts):
    print(f"Cup length {value} appears {count} times")
```

---

## 🧠 Main Function

```python
compute_persistent_cup_length_ripser(k, ripser_result, distance_matrix_full, min_persistence, thresh=math.inf)
```

Computes a persistent cup-length diagram from Ripser output.

**Returns**:
- Matrix representing cup-length intervals
- Lists of birth and death times
- Cup-length-2 barcode intervals

See docstring inside `cup_length_utils.py` for full documentation.

---

 
 

## 🛠️ Requirements

- Python ≥ 3.7
- `ripser`
- `numpy`
- `scipy==1.10.1`

---

## 📬 Questions?

Feel free to open an issue or contact ekaterina.s.ivshina@gmail.com for questions or suggestions.

 
