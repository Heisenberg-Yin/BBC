# 🌟 Bucket-based Result Collector

**Bucket-based Result Collector (BBC)** is a **plug-and-play tool** designed to enhance the efficiency of existing **approximate nearest neighbor (ANN)** methods for **large-$k$ ANN queries**.  It can be seamlessly integrated with existing quantization-based methods to accelerate the collection and selection phases.

---

## 🚀 Overview

Bucket-based Result Collector introduces a bucket-based result buffer that serves as the top-k collector, along with two new re-ranking algorithms designed to accelerate the re-ranking process.

---

## 🧩 Implementations

This repository includes:
- **Baselines**
  - `IVF+PQ`
  - `IVF+RaBitQ`
- **BBC-enhanced versions**
  - `IVF+PQ+BBC`
  - `IVF+RaBitQ+BBC`

Each implementation demonstrates how BBC integrates with existing quantization-based AKNN indexes to improve efficiency.

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
# ⚡ Run IVF+RaBitQ+BBC
# ---------------------------------------------------
cd src/RaBitQ-improve/
bash script/search.sh
cd ../../

# ---------------------------------------------------
# ⚡ Run IVF+PQ+BBC
# ---------------------------------------------------
cd src/OPQ-improve/
bash search.sh
cd ../../
