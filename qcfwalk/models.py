"""Models and helper functions for various gene prioritization methods."""

from dataclasses import dataclass
from typing import Callable, Iterable, Dict, List, Optional, Tuple, Union
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, diags, vstack, hstack, eye
from scipy.sparse.linalg import expm_multiply
import scipy.sparse.linalg as spla
import qcfwalk.utils as ut

"""Models and helper functions for QCFwalk."""


def perturb_network(
    G: nx.Graph,
    remove_fraction: float = 0.0,
    add_fraction: float = 0.0,
    random_seed: Optional[int] = None,
    forbidden_added_edges: Optional[Iterable[Tuple[int, int]]] = None,
) -> Tuple[nx.Graph, int, int]:
    """Return a copy of ``G`` after uniform edge deletion and non-edge addition.

    Fractions are relative to the number of edges in the unperturbed graph.
    Added edges are sampled uniformly from valid non-edges without self-loops,
    duplicate edges, replacement, or ``forbidden_added_edges``. Node identities
    and attributes are kept so score arrays remain aligned with the original graph.
    """
    if not 0.0 <= remove_fraction <= 1.0:
        raise ValueError("remove_fraction must be between 0 and 1.")
    if add_fraction < 0.0:
        raise ValueError("add_fraction must be non-negative.")
    if G.is_directed() or G.is_multigraph():
        raise ValueError("perturb_network requires a simple undirected graph.")

    rng = np.random.default_rng(random_seed)
    perturbed = G.copy()
    forbidden = {frozenset(edge) for edge in (forbidden_added_edges or [])}
    original_edge_count = G.number_of_edges()
    remove_count = int(round(remove_fraction * original_edge_count))
    add_count = int(round(add_fraction * original_edge_count))

    if remove_count:
        edge_indices = rng.choice(original_edge_count, size=remove_count, replace=False)
        edges = list(G.edges())
        perturbed.remove_edges_from([edges[index] for index in edge_indices])

    node_count = perturbed.number_of_nodes()
    available_nonedges = node_count * (node_count - 1) // 2 - perturbed.number_of_edges()
    if add_count > available_nonedges:
        raise ValueError(
            f"Cannot add {add_count} noise edges: only {available_nonedges} non-edges are available."
        )
    if add_count:
        # PPI networks are sparse but can have tens of thousands of genes;
        # materializing nx.non_edges() would require quadratic memory.
        nodes = np.asarray(list(perturbed.nodes()), dtype=object)
        added_edges = []
        added_keys = set()
        max_attempts = max(1000, add_count * 100)
        attempts = 0
        while len(added_edges) < add_count:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Unable to sample enough allowed noise edges; reduce the fraction or relax exclusions."
                )
            first, second = rng.choice(nodes, size=2, replace=False)
            edge_key = frozenset((first, second))
            if (
                perturbed.has_edge(first, second)
                or edge_key in added_keys
                or edge_key in forbidden
            ):
                continue
            added_edges.append((first, second))
            added_keys.add(edge_key)
        perturbed.add_edges_from(added_edges)

    return perturbed, remove_count, add_count


def degree_preserving_rewire(
    G: nx.Graph,
    rewire_fraction: float,
    random_seed: Optional[int] = None,
    forbidden_new_edges: Optional[Iterable[Tuple[int, int]]] = None,
) -> Tuple[nx.Graph, int]:
    """Randomly swap edge endpoints while preserving every node's degree.

    Each successful swap replaces ``(a, b), (c, d)`` with either
    ``(a, d), (c, b)`` or ``(a, c), (b, d)``. Self-loops, duplicate edges, and
    newly created forbidden edges are rejected. Existing forbidden edges remain
    untouched: the exclusion prevents artificial train-to-validation shortcuts.
    """
    if not 0.0 <= rewire_fraction <= 1.0:
        raise ValueError("rewire_fraction must be between 0 and 1.")
    if G.is_directed() or G.is_multigraph():
        raise ValueError("degree_preserving_rewire requires a simple undirected graph.")

    perturbed = G.copy()
    target_swaps = int(round(rewire_fraction * G.number_of_edges()))
    if target_swaps == 0:
        return perturbed, 0

    rng = np.random.default_rng(random_seed)
    forbidden = {frozenset(edge) for edge in (forbidden_new_edges or [])}
    edges = list(perturbed.edges())
    swaps = 0
    attempts = 0
    max_attempts = max(1000, target_swaps * 100)
    while swaps < target_swaps and attempts < max_attempts:
        attempts += 1
        first_index, second_index = rng.choice(len(edges), size=2, replace=False)
        first_edge = edges[first_index]
        second_edge = edges[second_index]
        a, b = first_edge
        c, d = second_edge
        if len({a, b, c, d}) != 4:
            continue
        new_edges = ((a, d), (c, b)) if rng.integers(2) else ((a, c), (b, d))
        new_keys = {frozenset(edge) for edge in new_edges}
        if len(new_keys) != 2 or any(len(edge) != 2 for edge in new_keys):
            continue
        if any(key in forbidden for key in new_keys):
            continue
        if any(perturbed.has_edge(*edge) for edge in new_edges):
            continue
        perturbed.remove_edges_from((first_edge, second_edge))
        perturbed.add_edges_from(new_edges)
        edges[first_index], edges[second_index] = new_edges
        swaps += 1

    if swaps != target_swaps:
        raise RuntimeError(
            f"Completed {swaps} of {target_swaps} requested degree-preserving swaps. "
            "Reduce the rewire fraction or relax exclusions."
        )
    return perturbed, swaps
def qa_score(
    G: nx.Graph,
    seed_list: List,
    t: float,
    H: csr_matrix,
    diag: Union[float, None] = None,
) -> np.ndarray:
    """Calculate quantum walk scores.

    Args:
    ----
    G: Graph that the walk occures on.
    seed_list: List of seed nodes.
    t: Time for which the walk lasts.
    H: Matrix to use as Hamiltonian.
    diag: How to set diagonals of the Hamiltonian.

    Returns:
    -------
    Array containing scores for each node in G.

    """
    n = G.number_of_nodes()
    n_seeds = len(seed_list)
    if isinstance(diag, (float, int)):
        # Construct sparse nxn matrix, with values `diag` at
        # entry (seed, seed), for each seed in seed_list:
        D = csr_matrix(([diag] * n_seeds, (seed_list, seed_list)), shape=(n, n))
        H += D
    # Trick for doing "quantum expm_multiply":
    Z = np.zeros((n, n_seeds), dtype=int)
    Z[seed_list, np.arange(n_seeds)] = 1  # each column corresponds to a seed
    res = expm_multiply(-1j * t * H, Z)
    res = np.abs(res) ** 2
    return res.sum(axis=1)


def ctqw_score(
        G: nx.Graph,
        seed_list: List[int],
        t: float,
        H: csr_matrix,
        diag: Union[float, None] = None,
) -> np.ndarray:
    """高效计算连续时间量子行走（CTQW）概率分布。

    基于哈密顿量进行酉演化，采用单列初始叠加态向量优化 expm_multiply 的执行速度。
    """
    n = G.number_of_nodes()
    if not seed_list:
        return np.zeros(n)

    # 1. 安全处理对角线哈密顿量修正
    if isinstance(diag, (float, int)):
        n_seeds = len(seed_list)
        D = csr_matrix(([diag] * n_seeds, (seed_list, seed_list)), shape=(n, n))
        H_eff = H + D  # 产生新矩阵，保护原 H 矩阵不被污染
    else:
        H_eff = H

    # 2. 构建初始量子叠加态
    # 量子态的初始叠加态振幅 (复数绝对值的平方和为 1)
    q_val = 1.0 / np.sqrt(len(seed_list))
    psi_0 = csr_matrix(([q_val] * len(seed_list), (seed_list, [0] * len(seed_list))), shape=(n, 1))

    # 3. 计算连续时间量子行走（CTQW）
    quantum_state = spla.expm_multiply(-1j * t * H_eff, psi_0)
    quantum_prob = np.asarray(np.abs(quantum_state.toarray()) ** 2).flatten()

    return quantum_prob


def ctrw_score(
        G: nx.Graph,
        seed_list: List[int],
        t: float,
        L: csr_matrix,
        restart_rate: float = 0.0,
) -> np.ndarray:
    """高效计算连续时间随机行走（CTRW）概率分布（支持连续时间重启特征）。

    通过增广矩阵技巧，在 O(N) 级别的时间内精确求解连续时间 RWR 的解析积分。
    """
    n = G.number_of_nodes()
    if not seed_list:
        return np.zeros(n)

    # 1. 构建经典态的初始概率分布 (总和为 1)
    c_val = 1.0 / len(seed_list)
    p_0 = csr_matrix(([c_val] * len(seed_list), (seed_list, [0] * len(seed_list))), shape=(n, 1))

    # 2. 计算带有重启特征的连续时间经典游走
    if restart_rate > 0:
        # 运用增广矩阵 (Augmented Matrix) 技巧
        # M = [ -(L + rI)   r * p_0 ]
        #     [     0          0    ]
        A = -(L + restart_rate * eye(n, format='csr'))
        B = restart_rate * p_0
        bottom_row = csr_matrix((1, n + 1))
        M = vstack([hstack([A, B]), bottom_row], format='csr')

        # 初始增广向量 v = [p_0; 1]
        v_0 = vstack([p_0, csr_matrix([[1.0]])], format='csr')

        # 求解状态演化，并提取前 n 个元素作为经典概率
        augmented_state = spla.expm_multiply(M * t, v_0)
        classical_prob = np.asarray(augmented_state[:-1].toarray()).flatten()
    else:
        # 退化为无重启的普通图热扩散
        classical_state = spla.expm_multiply(-t * L, p_0)
        classical_prob = np.asarray(classical_state.toarray()).flatten()

    return classical_prob

def build_renormalized_hamiltonian(G: nx.Graph, alpha: float = 0.5) -> csr_matrix:
    """
    构建用于量子游走的重正化哈密顿量 (Renormalized Hamiltonian)。

    公式: H_eff = D^(-alpha) * A * D^(-alpha)

    当 alpha = 0.5 时，等价于对称归一化拉普拉斯矩阵 (去除单位阵 I 的版本)。
    当 alpha = 0.0 时，退化为原始邻接矩阵 A。
    通过微调 alpha (如 0.2 ~ 0.8)，可以有效控制 Hub 节点对量子态的“引力”。

    Args:
    ----
    G: NetworkX 图对象
    alpha: 节点度的缩放指数

    Returns:
    -------
    稀疏矩阵格式的哈密顿量 H_eff
    """
    # 1. 提取稀疏邻接矩阵 (保持 CSR 格式)
    nodes = list(G.nodes())
    A = nx.adjacency_matrix(G, nodelist=nodes)

    if alpha == 0.0:
        return A

    # 2. 计算节点的度
    degrees = np.array([d for _, d in G.degree(nodes)])

    # 3. 计算 D^(-alpha)
    # 使用 np.errstate 避免图中存在孤立节点 (度为0) 时的除零警告
    with np.errstate(divide="ignore", invalid="ignore"):
        D_inv_alpha = np.power(degrees, -float(alpha))

    # 将孤立节点的 inf 或 nan 替换为 0
    D_inv_alpha[np.isinf(D_inv_alpha) | np.isnan(D_inv_alpha)] = 0.0

    # 4. 构建对角稀疏矩阵
    D_inv_alpha_matrix = diags(D_inv_alpha, format="csr")

    # 5. 矩阵乘法构建 H_eff (稀疏 @ 稀疏 @ 稀疏，速度极快)
    H_eff = D_inv_alpha_matrix @ A @ D_inv_alpha_matrix

    return H_eff

def qrw_score(
    G: nx.Graph,
    seed_list: List[int],
    t: float,
    L: csr_matrix,
    H: csr_matrix,
    diag: Union[float, None] = None,
    omega: Optional[float] = 0.5,
    restart_rate: float = 0.0,
    phase_mode: Union[str, np.ndarray, list] = "constant",
    random_seed: Optional[int] = None,
    score_permutations: int = 0,
    permutation_batch_size: int = 32,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """高效计算量子随机行走（QRW）评分（已融合连续时间重启特征）。

    When ``score_permutations`` is positive, additionally return one-sided
    empirical Score p-values from equal-size random seed sets. The observed
    QRW operators are reused and batches of seed initial states are propagated
    together instead of computing a matrix exponential per permutation.
    """
    n = G.number_of_nodes()
    if not seed_list:
        scores = np.zeros(n)
        return (scores, np.ones(n)) if score_permutations else scores
    if score_permutations < 0:
        raise ValueError("score_permutations must be non-negative.")
    if permutation_batch_size < 1:
        raise ValueError("permutation_batch_size must be at least 1.")
    if score_permutations and random_seed is None:
        raise ValueError("random_seed is required when computing Score p-values.")

    n_seeds = len(seed_list)

    # 1. 哈密顿量修正... (保持你原有的代码不变)
    if isinstance(diag, (float, int)):
        D = csr_matrix(([diag] * n_seeds, (seed_list, seed_list)), shape=(n, n))
        H_eff = H + D
    else:
        H_eff = H

    # 2. 【核心优化】引入复数相位构建初始量子态
    # 基础振幅大小
    base_amplitude = 1.0 / np.sqrt(n_seeds)

    # 确定相位角 theta
    if phase_mode == "constant":
        # 退化为传统版本：所有相位角为 0 (完全是实数 1+0j)
        theta = np.zeros(n_seeds)
    elif phase_mode == "random":
        # 随机相位：在 [0, 2π) 之间为每个 seed 赋予随机相位
        # 【核心修改】：使用局部随机状态生成器
        rng = np.random.default_rng(random_seed)
        theta = rng.uniform(0, 2 * np.pi, n_seeds)
    elif isinstance(phase_mode, (list, np.ndarray)):
        # 自定义相位：例如基于上游多组学特征传递过来的正负向信号
        theta = np.asarray(phase_mode)
        if len(theta) != n_seeds:
            raise ValueError("The length of custom phases must match the number of seeds.")
    else:
        raise ValueError(f"Unknown phase_mode: {phase_mode}")

    # 使用欧拉公式生成复数振幅: A * e^(i*theta)
    q_vals = base_amplitude * np.exp(1j * theta)

    # 构建稀疏初始量子态矩阵，必须显式指定 dtype=complex
    psi_0 = csr_matrix((q_vals, (seed_list, [0] * n_seeds)), shape=(n, 1), dtype=complex)

    # 构建经典的初始概率分布 (保持全为正实数，不受复数相位影响)
    c_val = 1.0 / n_seeds
    p_0 = csr_matrix(([c_val] * n_seeds, (seed_list, [0] * n_seeds)), shape=(n, 1))

    # 3. 计算连续时间量子行走（CTQW）部分... (后续逻辑保持你原有代码不变)
    quantum_state = spla.expm_multiply(-1j * t * H_eff, psi_0)
    quantum_prob = np.asarray(np.abs(quantum_state.toarray()) ** 2).flatten()

    # 4. 计算带有重启特征的连续时间经典游走（CT-RWR）部分
    if restart_rate > 0:
        # 运用增广矩阵 (Augmented Matrix) 技巧，将包含积分的微分方程转化为单一矩阵指数
        # M = [ -(L + rI)   r * p_0 ]
        #     [     0          0    ]
        A = -(L + restart_rate * eye(n, format='csr'))
        B = restart_rate * p_0
        bottom_row = csr_matrix((1, n + 1))
        M = vstack([hstack([A, B]), bottom_row], format='csr')

        # 初始增广向量 v = [p_0; 1]
        v_0 = vstack([p_0, csr_matrix([[1.0]])], format='csr')

        # 求解状态演化，并提取前 n 个元素作为经典概率
        augmented_state = spla.expm_multiply(M * t, v_0)
        classical_prob = np.asarray(augmented_state[:-1].toarray()).flatten()
    else:
        # 退化为无重启的普通图热扩散
        classical_state = spla.expm_multiply(-t * L, p_0)
        classical_prob = np.asarray(classical_state.toarray()).flatten()

    # === 👇 新增这一行：强制消除归一化拉普拉斯带来的概率膨胀 👇 ===
    classical_prob = classical_prob / np.sum(classical_prob)

    # 5. 基于物理逻辑的退相干权重融合
    if restart_rate > 0:
        # 重启率 r 表示对量子态的泊松测量/坍缩
        # 物理意义：重启(测量)会导致量子游走者发生波函数坍缩并转化为经典状态。
        # 纯量子态的存活概率为 e^{-rt}。被测量失去相干性的量子态部分，将完美汇入经典 RWR 的分布池中。
        survive_prob = np.exp(-restart_rate * t)
        eff_omega = omega * survive_prob
        eff_classical = (1 - omega) + omega * (1 - survive_prob)
        res = eff_omega * quantum_prob + eff_classical * classical_prob
    else:
        res = omega * quantum_prob + (1 - omega) * classical_prob
        # === 👇 插入这三行验证代码 👇 ===
        # print(f"🔎 验证 -> Quantum Prob 总和: {np.sum(quantum_prob):.8f}")
        # print(f"🔎 验证 -> Classical Prob 总和: {np.sum(classical_prob):.8f}")
        # print(f"🔎 验证 -> 最终融合 Res 总和: {np.sum(res):.8f}")
    if not score_permutations:
        return res

    score_pvalues = _seed_permutation_score_pvalues(
        observed_scores=res,
        seed_count=n_seeds,
        t=t,
        L=L,
        H_eff=H_eff,
        omega=omega,
        restart_rate=restart_rate,
        permutations=score_permutations,
        random_seed=random_seed,
        batch_size=permutation_batch_size,
    )
    return res, score_pvalues


def _seed_permutation_score_pvalues(
    observed_scores: np.ndarray,
    seed_count: int,
    t: float,
    L: csr_matrix,
    H_eff: csr_matrix,
    omega: float,
    restart_rate: float,
    permutations: int,
    random_seed: int,
    batch_size: int,
) -> np.ndarray:
    """Calculate Score p-values from batched random seed initial states."""
    n = len(observed_scores)
    rng = np.random.default_rng(random_seed)
    exceedances = np.zeros(n, dtype=np.int64)
    quantum_operator = -1j * t * H_eff

    if restart_rate > 0:
        diffusion_operator = -(L + restart_rate * eye(n, format="csr"))
        restart_operator = vstack(
            [
                hstack([diffusion_operator, restart_rate * eye(n, format="csr")]),
                csr_matrix((n, 2 * n)),
            ],
            format="csr",
        )
        survival_probability = np.exp(-restart_rate * t)
        quantum_weight = omega * survival_probability
        classical_weight = (1 - omega) + omega * (1 - survival_probability)
    else:
        classical_operator = -t * L
        quantum_weight = omega
        classical_weight = 1 - omega

    for start in range(0, permutations, batch_size):
        batch_count = min(batch_size, permutations - start)
        random_seeds = np.empty((seed_count, batch_count), dtype=np.intp)
        for column in range(batch_count):
            random_seeds[:, column] = rng.choice(n, size=seed_count, replace=False)

        row_indices = random_seeds.ravel(order="F")
        column_indices = np.repeat(np.arange(batch_count), seed_count)
        quantum_initial = csr_matrix(
            (
                np.full(seed_count * batch_count, 1.0 / np.sqrt(seed_count)),
                (row_indices, column_indices),
            ),
            shape=(n, batch_count),
            dtype=complex,
        )
        classical_initial = csr_matrix(
            (
                np.full(seed_count * batch_count, 1.0 / seed_count),
                (row_indices, column_indices),
            ),
            shape=(n, batch_count),
        )

        quantum_state = spla.expm_multiply(quantum_operator, quantum_initial)
        quantum_scores = np.abs(quantum_state.toarray()) ** 2
        if restart_rate > 0:
            restart_initial = vstack([classical_initial, classical_initial], format="csr")
            classical_scores = spla.expm_multiply(restart_operator * t, restart_initial)[:n]
        else:
            classical_scores = spla.expm_multiply(classical_operator, classical_initial)
        classical_scores = classical_scores.toarray()
        classical_scores /= classical_scores.sum(axis=0, keepdims=True)

        permuted_scores = quantum_weight * quantum_scores + classical_weight * classical_scores
        exceedances += np.count_nonzero(permuted_scores >= observed_scores[:, None], axis=1)

    return (exceedances + 1) / (permutations + 1)




def normalize_adjacency(G: nx.Graph, A: csr_matrix) -> csr_matrix:
    """Compute normalized adjacnecy matrix for RWR.

    Args:
    ----
    G: Graph to use.
    A: Sparse representation of the adjacency matrix.

    Returns:
    -------
    The normalized adjacency matrix, defined by D^(-1/2)AD^(-1/2).

    """
    degrees = np.array([d for _, d in G.degree(range(G.number_of_nodes()))])
    with np.errstate(divide="ignore"):  # Handle division by zero for isolated nodes
        D_inv_sqrt = np.power(degrees, -0.5)
    D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  # Replace inf with 0 for isolated nodes
    D_inv_sqrt_matrix = diags(D_inv_sqrt)
    return D_inv_sqrt_matrix @ A @ D_inv_sqrt_matrix


def rwr_score(
    G: nx.Graph,
    seed_list: List,
    normalized_adjacency: csr_matrix,
    return_prob: float = 0.75,
) -> np.ndarray:
    """Score nodes based on random walk with restart.

    Modified from
    https://github.com/mims-harvard/pathways/blob/master/prediction/randomWalk.py

    Args:
    ----
    G: Graph to use.
    seed_list: List of nodes that are seeds.
    return_prob: Probability of return to seed nodes.
    normalized_adjacency: D^{-1/2} A D^{-1/2}.

    Returns:
    -------
    Array containing scores for each node in G.

    """
    # Generate the train_seed_mask
    train_seed_mask = ut.seed_list_to_mask(seed_list, G.number_of_nodes())
    assoc_gene_vector = train_seed_mask
    ratio = return_prob
    convergence_metric = 1
    p0 = assoc_gene_vector / np.sum(assoc_gene_vector)
    old_vector = p0
    while convergence_metric > 1e-6:
        new_vector = (1 - ratio) * normalized_adjacency.dot(old_vector) + ratio * p0
        convergence_metric = np.linalg.norm(new_vector - old_vector)
        old_vector = np.copy(new_vector)
    return old_vector * (1 - assoc_gene_vector)



@dataclass(frozen=True)
class Model:
    """Holder for models containing some other useful data."""

    score_function: Callable  # function that returns scores
    name: str  # name of the model
    arguments: Dict  # arguments for the model