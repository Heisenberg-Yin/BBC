# 🌟 BRCollector

**BRCollector (BRC)** is a **plug-and-play tool** designed to enhance the efficiency of existing **approximate k-nearest neighbor (AKNN)** methods for **large-$k$ ANN queries**.  It can be seamlessly integrated with existing quantization-based methods to accelerate the collection and selection phases.

---

## 🚀 Overview

BRCollector introduces a bucket-based result buffer that serves as the top-k collector, along with two new re-ranking algorithms designed to accelerate the re-ranking process.

---

## 🧩 Implementations

This repository includes:
- **Baselines**
  - `IVF+PQ`
  - `IVF+RaBitQ`
- **BRC-enhanced versions**
  - `IVF+PQ+BRC`
  - `IVF+RaBitQ+BRC`

Each implementation demonstrates how BRC integrates with existing quantization-based AKNN indexes to improve efficiency.

---

## 📂 Datasets

The datasets used in our experiments can be downloaded from the **public sources referenced in the paper**. (See the paper’s experiment section for detailed download links.)

---

## 🛠️ Usage

```bash
# Clone the repository
# Clone the repository

# ---------------------------------------------------
# 🧩 Run IVF+RaBitQ
# (Implementation adapted from the official RaBitQ repo:
#  https://github.com/gaoj0017/RaBitQ)
# ---------------------------------------------------
python data/ivf.py
python data/rabitq.py
cd src/RaBitQ/
bash script/index.sh
bash script/search.sh
cd ../../

# ---------------------------------------------------
# 🧠 Run IVF+PQ
# ---------------------------------------------------
python data/ivf.py
python data/faiss_opq_index.py
cd src/OPQ/
bash script/index.sh
bash script/search.sh
cd ../../

# ---------------------------------------------------
# ⚡ Run IVF+RaBitQ+QR
# ---------------------------------------------------
cd src/RaBitQ-improve/
bash script/search.sh
cd ../../

# ---------------------------------------------------
# ⚡ Run IVF+PQ+QR
# ---------------------------------------------------
cd src/OPQ-improve/
bash search.sh
cd ../../
