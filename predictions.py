import argparse
from pathlib import Path
from typing import Dict
import networkx as nx
import numpy as np
import pandas as pd
import qcfwalk.data_process as dt
import qcfwalk.models as md
import qcfwalk.utils as ut


def add_mutation_frequency_ranks(results_df: pd.DataFrame, disease: str) -> pd.DataFrame:
    """Add sample-prevalence mutation ranks and their difference from QRW ranks.

    Only genes already present in ``results_df`` are retained. Gene names are
    stripped before matching to avoid mismatches caused by surrounding spaces.
    Higher ``%_SAMPLES_COHORT`` values receive better (smaller) mutation ranks.
    The column stores proportions, so 0.05 means 5% of cohort samples.
    Genes missing mutation-frequency values are assigned zero and the largest rank.
    """
    mutation_file = Path("data/TCGA_cancer_type_mut") / f"{disease}.txt"
    if not mutation_file.exists():
        raise FileNotFoundError(f"Mutation file not found: {mutation_file}")

    mutation_df = pd.read_csv(
        mutation_file,
        sep="\t",
        usecols=["gene_name", "%_SAMPLES_COHORT"],
        dtype={"gene_name": "string"},
    )
    mutation_df["gene_name"] = mutation_df["gene_name"].str.strip()
    mutation_df["%_SAMPLES_COHORT"] = pd.to_numeric(
        mutation_df["%_SAMPLES_COHORT"], errors="raise"
    )
    if not mutation_df["%_SAMPLES_COHORT"].between(0, 1).all():
        raise ValueError(
            f"%_SAMPLES_COHORT must be a proportion in [0, 1] in {mutation_file}."
        )
    if mutation_df["gene_name"].duplicated().any():
        raise ValueError(f"Duplicate gene_name values found in {mutation_file}.")

    results_df = results_df.copy()
    results_df["gene_name"] = results_df["gene_name"].astype("string").str.strip()
    results_df = results_df.merge(
        mutation_df, on="gene_name", how="left", validate="many_to_one"
    )
    missing_mutation = results_df["%_SAMPLES_COHORT"].isna()
    results_df["%_SAMPLES_COHORT"] = results_df["%_SAMPLES_COHORT"].fillna(0)
    results_df["Rank_mutation"] = (
        results_df["%_SAMPLES_COHORT"].rank(method="min", ascending=False).astype("Int64")
    )
    results_df.loc[missing_mutation, "Rank_mutation"] = len(results_df)
    results_df["Rank_improvement"] = results_df["Rank_mutation"] - results_df["Rank"]
    return results_df


def rank_improvement_pvalues(
    qrw_ranks: np.ndarray,
    mutation_ranks: np.ndarray,
    permutations: int,
    random_seed: int,
) -> np.ndarray:
    """Calculate one-sided p-values for positive improvements under random QRW ranks."""
    if permutations < 1:
        raise ValueError("rank_permutations must be at least 1.")

    rng = np.random.default_rng(random_seed)
    observed_improvement = mutation_ranks - qrw_ranks
    exceedances = np.zeros(len(qrw_ranks), dtype=np.int64)
    for _ in range(permutations):
        permuted_ranks = rng.permutation(qrw_ranks)
        exceedances += mutation_ranks - permuted_ranks >= observed_improvement

    pvalues = (exceedances + 1) / (permutations + 1)
    pvalues[observed_improvement <= 0] = 1.0
    return pvalues


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description="QRW main program")
    parser.add_argument("-o", "--topn", type=int, default=20000) # Only top genes are exported in the result file.
    parser.add_argument("-d", "--disease", type=str, default="BRCA")
    parser.add_argument("-n", "--network", type=str, default="String")
    parser.add_argument("-m", "--omega", type=int, default=0.7)
    parser.add_argument("-t", "--time", type=int, default=3)
    parser.add_argument("-e", "--seed", type=int, default=2)
    parser.add_argument("-g", "--diag", type=int, default=5)
    parser.add_argument("-s", "--restart", type=int, default=0.1)
    parser.add_argument(
        "-p",
        "--permutations",
        type=int,
        default=10000,
        help="Number of random seed-set permutations for Score empirical p-values.",
    )
    parser.add_argument(
        "--rank-permutations",
        type=int,
        default=10000,
        help="Number of random QRW-rank improvement permutations for Rank_improvement_Pvalue.",
    )
    args = parser.parse_args()
    return vars(args)

def main() -> None:
    params = parse_args()
    network = params["network"]
    disease = params["disease"]
    topn = params["topn"]
    seed = params["seed"]
    omega = params["omega"]
    t = params["time"]
    diag = params["diag"]
    restart = params["restart"]
    permutations = params["permutations"]
    rank_permutations = params["rank_permutations"]

    print(f"\n{'='*30}\n🌐 Loading and processing: {network}\n{'='*30}")
    try:
        G, code_dict, seeds_by_disease = dt.load_dataset(
            disease,
            network,
            dt.FilterGCC.TRUE,
        )
    except Exception as e:
        print(f"⚠️ Cannot load {network} PPI with {disease}. Skipping. Error: {e}")


    # 获取当前疾病对应的两套独立基因集
    train_genes = seeds_by_disease.get(disease, [])

    if not train_genes:
        print(f"ℹ️ {network} PPI with {disease} lacks seed genes, skpping.")
        # sys.exit()
    print(
        f"Task activated. Network: {network} | Disease type: {disease} | Training set: {len(train_genes)}")

    if disease not in seeds_by_disease:
        e = f"{disease} is not found in the cancer type set."
        raise ValueError(e)
    inv_code_dict = ut.inv_code_dict(code_dict)
    n = G.number_of_nodes()
    nl = range(n)
    L = nx.normalized_laplacian_matrix(G, nodelist=nl)
    H_scaled = md.build_renormalized_hamiltonian(G, alpha=0.5)

    seeds = seeds_by_disease[disease]
    scores, score_pvalues = md.qrw_score(
        G,
        seeds,
        t=float(t),
        L=L,
        H=H_scaled,
        diag=float(diag),
        omega=float(omega),
        restart_rate=restart,
        phase_mode="random",
        random_seed=seed,
        score_permutations=permutations,
    )

    # ---------------------------------------------------------
    # 【极简逻辑】：保留训练集，全量节点参与排序
    # ---------------------------------------------------------
    # 此时 scores 数组未被截断，其内部的索引直接对应图中的节点绝对 ID
    current_topn = min(topn, n)

    ind = np.argpartition(scores, -current_topn)[-current_topn:]
    top_N_global = ind[np.argsort(scores[ind])]

    # ---------------------------------------------------------
    # 纯离线本地文件 Gene ID 转换 Gene Symbol
    # ---------------------------------------------------------
    print("🔄 正在从本地网络文件提取 Gene ID 到 Gene Symbol 的映射关系...")
    network_file = Path(f"data/{network}/{disease}.txt")
    id_to_symbol = {}

    if network_file.exists():
        mapping_df = pd.read_csv(
            network_file,
            sep='\t',
            low_memory=False,
            usecols=['geneid_1', 'gene_1', 'geneid_2', 'gene_2']
        )
        map1 = dict(zip(mapping_df['geneid_1'], mapping_df['gene_1']))
        map2 = dict(zip(mapping_df['geneid_2'], mapping_df['gene_2']))
        id_to_symbol.update(map1)
        id_to_symbol.update(map2)
        print(f"✅ 成功加载本地映射字典，共提取 {len(id_to_symbol)} 个基因对应关系。")
    else:
        print(f"⚠️ 找不到本地网络文件 {network_file}，将无法进行名称转换。")

    # ---------------------------------------------------------
    # 构建结果 DataFrame
    # ---------------------------------------------------------
    rows = []

    # 提前生成训练集集合，提升查找速度（用于在输出中标记该基因是否为种子基因）
    train_genes_set = set(train_genes)

    for i in range(current_topn):
        node_id = top_N_global[i]
        raw_id = inv_code_dict[node_id]
        gene_name = id_to_symbol.get(raw_id, raw_id)
        rank = current_topn - i
        deg = G.degree(node_id)
        # 提取当前节点的概率值 (Score)
        node_prob = scores[node_id]

        # 可选优化：在输出结果中新增一列 "Is_Seed"，明确标识它是不是训练集里的已知致病基因
        is_seed = "Yes" if node_id in train_genes_set else "No"

        # 将概率值纳入当前行中
        score_pvalue = score_pvalues[node_id]
        rows.append([gene_name, raw_id, network, disease, rank, node_prob, score_pvalue, deg, is_seed])

    # 在 columns 中新增 "Score" 列
    results_df = pd.DataFrame(
        rows, columns=["gene_name", "entrez_id", "Network", "Disease", "Rank", "Score", "Score_Pvalue", "Degree", "Is_Seed"]
    )

    results_df = results_df.sort_values(by="Rank", ascending=True)
    results_df = add_mutation_frequency_ranks(results_df, disease)
    results_df["Rank_improvement_Pvalue"] = rank_improvement_pvalues(
        results_df["Rank"].to_numpy(),
        results_df["Rank_mutation"].to_numpy(),
        rank_permutations,
        seed,
    )

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_file = out_dir / f"QRW-{network}-{disease}.csv"
    results_df.to_csv(preds_file, index=False)

    print(f"✅ Predictions saved to {preds_file}.")


if __name__ == "__main__":
    main()