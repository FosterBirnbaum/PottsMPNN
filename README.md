# PottsMPNN

Code for running PottsMPNN to generate protein sequences and predict mutational effects (energies).

## 1. Installation

We recommend using **Conda** to manage the environment and **pip** to install dependencies.

### Step 1: Create a Conda Environment
Create a clean environment with Python 3.10:
```bash
conda create -n PottsMPNN python=3.10
conda activate PottsMPNN

```

### Step 2: Install PyTorch

Install the version of PyTorch compatible with your system (CUDA/GPU recommended). Visit the [official PyTorch installation page](https://pytorch.org/get-started/locally/) to get the correct command. For example:

**Linux with CUDA 12.1:**

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

```

**Mac (CPU/MPS):**

```bash
pip install torch torchvision torchaudio

```

### Step 3: Install Dependencies

Install the required Python packages using the provided requirements file:

```bash
pip install -r requirements.txt

```

---

## 2. Running PottsMPNN

There are three ways to run the model, ranging from easiest (cloud-based) to advanced (command-line).

### Level 1: Google Colab (Easiest)

Run the model entirely in your browser using Google's free GPU resources. No local installation required.

* **Sequence Generation:** [Use this file to design new sequences for a backbone](https://colab.research.google.com/drive/1Jx447uZHwi_pvLbzYtdL961vsatAjlnd?usp=sharing)
* **Energy Prediction:** [Use this file to predict ΔΔG values for specific mutations.](https://colab.research.google.com/drive/1nAWcQXW_GQkyyN0X2s0G68w-8y0wDbpx?usp=sharing)

### Level 2: Local Jupyter Notebooks

If you have set up the installation environment above, you can run the interactive notebooks locally. This allows for easier file management and faster execution on local GPUs.

1. Start Jupyter:
```bash
jupyter notebook

```


2. Open `sample_seqs.ipynb` for sequence generation.
3. Open `energy_prediction.ipynb` for mutational scoring.

### Level 3: Command Line Interface (Advanced)

For batch processing or integration into pipelines, run the Python scripts directly using a YAML configuration file.

**Predicting Energies:**

```bash
python energy_prediction.py --config inputs/example_energy_config.yaml

```

**Generating Sequences:**

```bash
python sample_seqs.py --config inputs/example_sample_config.yaml

```

---

## 3. Configuration Options

Both pipelines use a configuration dictionary (or YAML file) to control the model. Example configurations can be found in the `inputs/` directory.


### Sequence Generation Options (`sample_seqs`)

Used when designing new sequences for a given backbone.

### Data input

* **`input_dir`**: Path to directory containing structures in .pdb format
* **`input_list`**: Path to .txt file identifying structure for which to sample sequences. Chain information (i.e., which chains to design and which are visible to the model) can be specified using the `'|'` token to distinguish between groups of designed and visible chains, which in a group should be split using the `':'` token. For example, a line in the .txt file of `prot|A:B:C|D:E` indicates that chains A, B, and C should be designed and chains D and E should be visible for structure prot.pdb. You can put the same structure on multiple lines so long as there are different combinations of designed and visible chains. If no chain information is specified, all chains will be designed.
* **`chain_dict_json`**: Path to a .json file identifying which chains should be designed and which are visible. For example, an entry in the .json of `prot: [[A, B, C], [D, E]]` indicates that chains A, B, and C should be designed and chains D and E should be visible for structure prot.pdb. Providing a .json file will overwrite information in **`input_list`**.

#### Sampling & Optimization

* **`inference.num_samples`**: (int) Number of sequences to generate per structure. Must be `1` if running optimization.
* **`inference.temperature`**: (float) Sampling temperature. Lower values (e.g., 0.1) produce more probable sequences; higher values (e.g., 1.0) add diversity.
* **`inference.noise`**: (float) Sampling temperature. Lower values (e.g., 0.1) produce more probable sequences; higher values (e.g., 1.0) add diversity.
* **`inference.optimization_mode`**: Optimization protocol to use:
* `"none"`: No optimization (just autoregressive sampling).
* `"potts"`: Optimizes sequence using Potts energy.
* `"nodes"`: Optimizes node features.

* **`inference.binding_energy_optimization`**: How to optimize using binding energies:
* `"none"`: Optimize stability only.
* `"both"`: Jointly optimize stability and binding affinity.
* `"only"`: Optimize binding affinity only.


#### Constraints & Biases

* **`inference.optimize_pdb`**: (bool) Optimize sequences found in the input `.pdb` files.
* **`inference.optimize_fasta`**: (bool) Optimize sequences found in an input `.fasta` file.
* **`inference.fixed_positions_json`**: Path to JSON defining 1-indexed positions to fix (keep as wildtype).
* **`inference.pssm_json`**: Path to JSON containing Position-Specific Scoring Matrix (bias per position).
* **`inference.omit_AA_json`**: Path to JSON defining amino acids to ban at specific positions.
* **`inference.bias_AA_json`**: Path to JSON defining global amino acid biases.
* **`inference.omit_AAs`**: List of amino acids to globally omit from design (e.g., `['C', 'W']`).

### Energy Prediction Options (`energy_prediction`)

Used when scoring specific mutations or calculating ΔΔG.

* **`input_list`**: Path to a `.txt` file containing the list of PDBs to process.
* **`mutant_fasta`** / **`mutant_csv`**: Path to input files defining the mutations to score.
* **`inference.ddG`**: (bool) If True, computes ΔΔG (mutant - wildtype). If False, outputs raw ΔG.
* **`inference.mean_norm`**: (bool) If True, centers predictions to mean=0.
* **`inference.filter`**: (bool) Filter output to only include mutations present in input files.
* **`inference.binding_energy_chains`**: (dict) Defines chains to separate for binding energy calculations (e.g., `{'1ABC': [['A'], ['B']]}`).


```

```
