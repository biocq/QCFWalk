# QCFWalk - Quantum-Classical Fusion Walk for Cancer-Gene Prioritization

QCFWalk integrates coherent multi-seed quantum propagation with classical continuous-time diffusion to prioritize cancer-associated genes in protein-protein interaction (PPI) networks. The `predictions.py` program generates gene rankings, empirical score statistics, and comparisons with cancer-specific mutation prevalence.

## Installation

1. Install Python and, optionally, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage an isolated environment.
2. Obtain `predictions.py`, the `qcfwalk` package, and the processed network, seed-gene, and mutation datasets. The script alone is insufficient: it imports `qcfwalk.data_process`, `qcfwalk.models`, and `qcfwalk.utils`.
3. Install the directly imported third-party libraries:

   ```bash
   python -m pip install networkx numpy pandas scipy
   ```
   or Create and activate the environment with:
   ```bash
   conda env create -f qcfwalk.yaml
   conda activate qcfwalk
   ```
4. Install the `qcfwalk` package using its repository's installation instructions, or make its source package importable from the project root. If the complete project includes a `pyproject.toml` or `setup.py`, an editable installation can be made from that root:

   ```bash
   python -m pip install -e .
   ```

   This command requires packaging metadata. No environment file, dependency lockfile, or `qcfwalk` source package accompanied the supplied script; any additional package dependencies must be installed from the complete project's dependency specification.

## Data Preparation

Run the program from the project root. Input and output paths are relative to the working directory.

```text
project/
├── predictions.py
├── qcfwalk/
│   ├── data_process.py
│   ├── models.py
│   └── utils.py
├── data/                       # Download from https://doi.org/10.6084/m9.figshare.34397397
│   ├── BioGRID/                # PPI by cancer type
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── CPDB/
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── IID/
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── IREF2015/
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── MULTINET/
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── PCNET/
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── String/
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├── TCGA_cancer_type_mut/    # Mutation rate by cancer type
│   │   ├── ACC.txt
│   │   ├── ...
│   │   └── UCS.txt
│   ├──  seed_gene.tsv
│   └──  test_gene.tsv
└── out/                         # Created automatically
```

Data fold should be placed by unzipping the downloaded data.zip file from https://doi.org/10.6084/m9.figshare.34397397.
### PPI Networks and Seed Genes

The dataset loader from data_process.py receives the disease name, network name. The script therefore requests the loader's giant-connected-component filtering mode (dt.FilterGCC). dt.FilterGCC value (True or False) does not impact the prediction results very much. Supported network and disease names depend on the loader and available data; the script defaults to `String` and `BRCA` and does not enumerate all supported combinations.

For gene-ID-to-symbol annotation, `data/<network>/<disease>.txt` must be tab-separated and contain:

```text
geneid_1    gene_1    geneid_2    gene_2
```

These are the columns read by the annotation step, not necessarily the loader's complete required schema. Gene identifiers should be consistent with the identifiers returned by the loader. If the local annotation file is absent, original identifiers are retained; network loading still requires valid loader inputs.

### Mutation Prevalence

`data/TCGA_cancer_type_mut/<disease>.txt` is required for every run and must be tab-separated with these columns:

```text
gene_name    %_SAMPLES_COHORT
```

`%_SAMPLES_COHORT` must contain numeric proportions between 0 and 1: `0.05` means 5% of cohort samples, not 0.05%. Gene names must be unique after surrounding whitespace is removed. Missing files, duplicate gene names, nonnumeric values, and values outside the permitted range cause the mutation-annotation step to fail.

## Usage

### Predictions

To generate predictions with the default settings:

```bash
python predictions.py
```

This requests up to 20,000 genes for BRCA using the String network and writes:

```text
out/QRW-String-BRCA.csv
```

To export the top 200 genes:

```bash
python predictions.py --network String --disease BRCA --topn 200
```

To specify the number of permutations and the random seed:

```bash
python predictions.py --network String --disease BRCA --topn 1000 --permutations 10000 --rank-permutations 10000 --seed 2
```

The number exported is capped at the number of retained network nodes. Seed genes are included and marked in the output. Runs using the same network and disease overwrite the same CSV, regardless of the parameter settings; retain separate copies when comparing runs. Runtime depends on network size and permutation counts.

For the available command-line arguments:

```bash
python predictions.py --help
```

### Parameters

| Argument | Short Form | Default | Description |
| --- | --- | --- | --- |
| `--topn` | `-o` | `20000` | Maximum number of genes exported; use a positive integer. |
| `--disease` | `-d` | `BRCA` | Disease or cancer code recognized by the dataset loader. |
| `--network` | `-n` | `String` | Network name recognized by the loader and matching the local data directory. |
| `--omega` | `-m` | `0.7` | Quantum mixing parameter passed to `qrw_score`. See the parsing limitation below. |
| `--time` | `-t` | `3` | Propagation time; explicit command-line values currently accept integers only. |
| `--seed` | `-e` | `2` | Random seed passed to the scoring routine and rank-permutation analysis. |
| `--diag` | `-g` | `5` | Diagonal-potential parameter passed to `qrw_score`; explicit values currently accept integers only. |
| `--restart` | `-s` | `0.1` | Classical restart rate passed to `qrw_score`. See the parsing limitation below. |
| `--permutations` | `-p` | `10000` | Random seed-set permutation count requested for empirical score statistics. |
| `--rank-permutations` | — | `10000` | Number of rank-permutation iterations; must be at least 1. |

**Current argument-parsing limitation:** `--omega` and `--restart` are declared with `type=int`, despite their floating-point defaults. Omitting these flags preserves the default values, but explicitly passing `--omega 0.7` or `--restart 0.1` raises an argument-parsing error. To expose continuous-valued settings, change their argument types to `float`; the same change is appropriate for `--time` and `--diag` if fractional values are required. The usage examples above work around this limitation by retaining the defaults.

## Prediction Workflow

1. Load the selected PPI network and disease seeds through `load_dataset`.
2. Construct the symmetric normalized Laplacian and a degree-renormalized Hamiltonian with `alpha=0.5`.
3. Call `qcfwalk.models.qrw_score` to calculate fusion scores and empirical score statistics. The current script explicitly uses `phase_mode="random"`; phase mode cannot be changed through its command-line arguments.
4. Select the highest-scoring nodes and assign ranks, with rank 1 corresponding to the highest score.
5. Recover gene symbols from the local network file and annotate network degree and seed status.
6. Match mutation prevalence, compute mutation ranks within the exported gene set, and calculate rank improvement.
7. Estimate one-sided rank-improvement statistics by randomly permuting the exported QCFWalk ranks and save the result as CSV.

The internal score formula, phase initialization, seed-null construction, and numerical implementation are defined in `qcfwalk.models`; they cannot be verified from this entry-point script alone.

## Output

Results are sorted by ascending `Rank` and saved to `out/QRW-<network>-<disease>.csv`. The `QRW` filename prefix is retained by the current implementation even though the method is referred to as QCFWalk.

| Column | Meaning |
| --- | --- |
| `gene_name` | Gene symbol from the local network mapping, or original identifier if unavailable. |
| `entrez_id` | Original node identifier returned by the inverse mapping; confirm its identifier system in the input data. |
| `Network` | Selected network name. |
| `Disease` | Selected disease or cancer code. |
| `Rank` | QCFWalk rank; smaller values indicate higher priority. |
| `Score` | Node score returned by `qrw_score`. |
| `Score_Pvalue` | Node-level empirical score statistic returned by `qrw_score`. |
| `Degree` | Degree in the processed network. |
| `Is_Seed` | `Yes` for supplied seed genes and `No` for other genes. |
| `%_SAMPLES_COHORT` | Mutation prevalence as a proportion; missing matches are replaced with zero. |
| `Rank_mutation` | Descending mutation-prevalence rank among exported genes. Observed ties use the minimum tied rank; missing mutation matches are assigned the number of exported genes. |
| `Rank_improvement` | `Rank_mutation - Rank`; positive values indicate higher priority under QCFWalk. |
| `Rank_improvement_Pvalue` | One-sided empirical rank-permutation statistic; nonpositive improvements are assigned 1. |

### Interpretation

- Remove rows with `Is_Seed == "Yes"` when constructing a list of non-seed candidates or evaluating recovery of held-out genes. A non-seed gene is not automatically novel relative to other cancer-gene databases.
- Positive rank improvement describes disagreement between network and mutation rankings. It does not establish low mutation prevalence, driver activity, or therapeutic selectivity.
- If applying the manuscript's low-frequency definition, use `0.01 <= %_SAMPLES_COHORT < 0.05` explicitly. The script does not create that category automatically.
- Mutation ranks and rank-permutation statistics depend on `--topn`, because they are calculated within the exported set. Keep that setting fixed for comparable analyses.
- Missing mutation matches are stored as zero without a separate missingness indicator. They cannot be distinguished from observed zero prevalence in the exported CSV alone.
- Rank-permutation statistics use `(exceedances + 1) / (permutations + 1)`. The CSV contains no multiple-testing-adjusted columns. Because the mutation rank cancels in the implemented exceedance comparison, positive-improvement statistics depend on the QCFWalk rank and the exported rank universe; they are not independent evidence of mutation-specific biological significance.
- The current entry point uses constant phases. Results should not be described as random-phase predictions unless the implementation is changed and rerun.

## Custom Networks and Diseases

Adapt `qcfwalk.data_process.load_dataset` to load the desired network and seed genes. Provide consistently mapped node identifiers, a supported disease key, the local network annotation file, and the required mutation-prevalence file. Arbitrary network paths and seed files cannot be supplied directly through the current CLI.

The present script always performs mutation annotation. Use for non-cancer diseases without mutation-prevalence data requires adapting that step; it is not supported by changing the disease flag alone.

## Troubleshooting

- **`ModuleNotFoundError: qcfwalk`**: install the complete package or run from a location where its source is importable.
- **Mutation file not found**: check `data/TCGA_cancer_type_mut/<disease>.txt` and run from the project root.
- **Invalid integer value for a decimal parameter**: retain the defaults or correct the argument types described above.
- **Dataset-loading or missing-seed errors**: check the loader's supported names and seed mappings. The current exception handler prints a loading warning but does not terminate, and the empty-seed branch also continues; subsequent errors can obscure the original problem.

## Citation

If you use this workflow, cite the QCFWalk manuscript, *QCFWalk: A quantum-classical fusion walk model for disease gene prioritization*, when a public citation is available. No publication DOI or software license was supplied with the script; this README does not assign either.
