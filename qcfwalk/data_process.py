"""Handle cancer-specific graph and gene sets."""

import csv
from enum import Flag
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, get_args
import networkx as nx
import pandas as pd

VALID_DATASETS = Literal["ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
                         "GBM", "HNSC",  "KIRC",  "LAML", "LGG", "LIHC", "LUAD", "LUSC", 
                         "OV", "PAAD", "PRAD", "READ", "SARC", "SKCM", "STAD",  "THYM", 
                         "UCEC", "UCS"]
valid_datasets: List[str] = list(get_args(VALID_DATASETS))

VALID_NETWORKS = Literal[
    "BioGRID", "String", "IID", "MULTINET", "CPDB", "PCNET", "IREF2015"
]

valid_networks: List[str] = list(get_args(VALID_NETWORKS))


class FilterGCC(Flag):
    """Whether to use the full PPI network or only the greatest connected component."""

    TRUE = True
    FALSE = False


def _build_graph(
    g_df: pd.DataFrame,
    filter_method: FilterGCC,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Turn pandas dataframe into graph.

    Args:
    ----
    g_df: Dataframe for the undirected graph with "source" and "target" columns.
    filter_method: Use the full network or only the greatest connected component.

    Returns:
    -------
    Networkx graph and code_dict which maps gene ids to graph nodes.

    """
    if filter_method == FilterGCC.TRUE:
        G_ = nx.from_pandas_edgelist(g_df)
        Gcc = sorted(nx.connected_components(G_), key=len, reverse=True)
        G0 = G_.subgraph(Gcc[0])
        gcc_nodes = list(G0.nodes())
        g_df = pd.DataFrame(g_df[g_df.source.isin(gcc_nodes)])
        g_df = pd.DataFrame(g_df[g_df.target.isin(gcc_nodes)])
    # Map gene ids to {0, 1, ..., num_nodes}:
    ids = pd.concat([g_df["source"], g_df["target"]])
    vals = ids.astype("category").cat.codes.to_numpy()
    code_dict = {}
    for gid, v in zip(ids, vals):
        code_dict[gid] = v
    g_df = g_df.replace({"source": code_dict})
    g_df = g_df.replace({"target": code_dict})
    G = nx.from_pandas_edgelist(g_df)
    return G, code_dict

def build_graph_biogrid(
    data_path: Path,
    filter_method: FilterGCC,
    network: str,
    cancertype: str,
) -> Tuple[nx.Graph, Dict[int, int]]:
    """Read csv file and convert it to a networkx graph.

    Args:
    ----
    data_path: Path to data directory.
    filter_method: Use the full network or only the greatest connected component.

    Returns:
    -------
    Networkx graph and code_dict which maps gene ids to graph nodes.
    """
    wl_df = pd.read_csv(
        f"../data/{network}/{cancertype}.txt", # >= python 3.6
        sep='\t',
        header=0, low_memory=False, usecols=['geneid_1','gene_1', 'geneid_2', 'gene_2']
    )
    wl_df.columns = ['source','gene_1', 'target', 'gene_2']
    G, code_dict = _build_graph(pd.DataFrame(wl_df), filter_method)
    return G, code_dict



def load_train_genes(
    node_list: List[int],
    code_dict: Dict[int, int],
    data_path: Path,
    cancertype: str,
) -> pd.DataFrame:
    """Read disease file and return a dataframe associating diseases and gene nodes.

    Args:
    ----
    node_list: List of nodes from the graph, ignored in this function.
    code_dict: Dictionary mapping gene ids to node labels.
    data_path: Path to the data directory.

    Returns:
    -------
    Pandas dataframe with "disease" and "gene" columns.

    """
    ot_df = pd.read_csv(
        data_path / "seed_gene.tsv",
        delimiter="\t",
        usecols=["gene", "symbol", "disease"]
    )
    ot_df = ot_df.drop_duplicates()
    
    ot_df["gene"] = ot_df.apply(lambda row: code_mapper(row, code_dict, "gene"), axis=1)
    ot_df = ot_df.loc[ot_df['disease'].isin([cancertype])]
    ot_df = ot_df[ot_df['gene'].notna()]
    ot_df["gene"] = ot_df["gene"].astype(int)

    return ot_df


def load_test_genes(
        node_list: List[int],
        code_dict: Dict[int, int],
        data_path: Path,
        cancertype: str,
) -> pd.DataFrame:
    """Read disease file and return a dataframe associating diseases and gene nodes.

    Args:
    ----
    node_list: List of nodes from the graph, ignored in this function.
    code_dict: Dictionary mapping gene ids to node labels.
    data_path: Path to the data directory.

    Returns:
    -------
    Pandas dataframe with "disease" and "gene" columns.

    """
    ot_df = pd.read_csv(
        data_path / "test_gene.tsv",
        delimiter="\t",
        usecols=["gene", "symbol", "disease"]
    )
    ot_df = ot_df.drop_duplicates()

    ot_df["gene"] = ot_df.apply(lambda row: code_mapper(row, code_dict, "gene"), axis=1)
    ot_df = ot_df.loc[ot_df['disease'].isin([cancertype])]
    ot_df = ot_df[ot_df['gene'].notna()]
    ot_df["gene"] = ot_df["gene"].astype(int)
    return ot_df


def code_mapper(
    row: pd.Series,
    code_dict: Dict[int, int],
    col_name: str,
) -> Optional[int]:
    """Apply code_dict to row of DataFrame to change entry in col_name.

    Args:
    ----
    row: Row object from the DataFrame
    code_dict: Dictionary that maps original gene names to node ids
    col_name: Name of the column holding the data
    """
    if int(row[col_name]) in code_dict:
        return code_dict[int(row[col_name])]
    return None


def get_cancer_genes(
    node_list: List,
    code_dict: Dict[int, int],
    data_path: Path,
    cancertype: str,
    method: Callable,
) -> Dict[str, List]:
    """Build dictionary mapping disease names to node labels.

    Args:
    ----
    node_list: List of nodes in the graph.
    code_dict: Maps gene ids to node ids.
    data_path: Path where data is stored.
    method: Method for reading the desired data set.

    Returns:
    -------
    Dictionary mapping disease names to node labels.

    """
    df_data = method(node_list, code_dict, data_path, cancertype)
    return df_data.groupby("disease")["gene"].agg(list).to_dict()


def load_dataset(
    disease_set: VALID_DATASETS,
    network: VALID_NETWORKS,
    filter_method: FilterGCC,
    
) -> Tuple[nx.Graph, Dict[int, int], Dict[str, List[int]]]:
    """Build network and disease data from raw files.

    Args:
    ----
    disease_set: Name of the data set.
    network: Name of the ppi network.
    filter_method: Whether or not to keep only the greatest connected compoenet.

    Returns:
    -------
    G: The PPI network.
    code_dict: Dictionary mapping gene ids to node values.
    disease_nodes_by_disease: Dictionary mapping disease names to a list of seed nodes.

    """
    if disease_set not in valid_datasets:
        e = f"{disease_set} not recognized. Must be one of {valid_datasets}"
        raise ValueError(e)
    if network not in valid_networks:
        e = f"{network} not recognized. Must be one of {valid_networks}"
        raise ValueError(e)
    data_path = Path("data")

    # Load the network:

    G, code_dict = build_graph_biogrid(data_path, cancertype= disease_set, network= network, filter_method=filter_method)

    # Load the cancer-type specific genes:
    traingenes_by_disease = get_cancer_genes(
        list(G.nodes()),
        code_dict,
        data_path,
        disease_set,
        load_train_genes
    )

    return G, code_dict, traingenes_by_disease


def load_dataset2(
        disease_set: VALID_DATASETS,
        network: VALID_NETWORKS,
        filter_method: FilterGCC,

) -> Tuple[nx.Graph, Dict[int, int], Dict[str, List[int]], Dict[str, List[int]]]:
    """Build network and disease data from raw files.

    Args:
    ----
    disease_set: Name of the data set.
    network: Name of the ppi network.
    filter_method: Whether or not to keep only the greatest connected compoenet.

    Returns:
    -------
    G: The PPI network.
    code_dict: Dictionary mapping gene ids to node values.
    disease_nodes_by_disease: Dictionary mapping disease names to a list of seed nodes.

    """
    if disease_set not in valid_datasets:
        e = f"{disease_set} not recognized. Must be one of {valid_datasets}"
        raise ValueError(e)
    if network not in valid_networks:
        e = f"{network} not recognized. Must be one of {valid_networks}"
        raise ValueError(e)
    data_path = Path("data")

    # Load the network:

    G, code_dict = build_graph_biogrid(data_path, cancertype=disease_set, network= network, filter_method=filter_method)

    # Load the cancer-type specific genes:
    traingenes_by_disease = get_cancer_genes(
        list(G.nodes()),
        code_dict,
        data_path,
        disease_set,
        load_train_genes
    )
    testgenes_by_disease = get_cancer_genes(
        list(G.nodes()),
        code_dict,
        data_path,
        disease_set,
        load_test_genes
    )

    return G, code_dict, traingenes_by_disease, testgenes_by_disease


def load_dataset_independent(
        disease_set: VALID_DATASETS,
        network: VALID_NETWORKS,
        filter_method: FilterGCC,

) -> Tuple[nx.Graph, Dict[int, int], Dict[str, List[int]]]:
    """Build network and disease data from raw files.

    Args:
    ----
    disease_set: Name of the data set.
    network: Name of the ppi network.
    filter_method: Whether or not to keep only the greatest connected compoenet.

    Returns:
    -------
    G: The PPI network.
    code_dict: Dictionary mapping gene ids to node values.
    disease_nodes_by_disease: Dictionary mapping disease names to a list of seed nodes.

    """
    if disease_set not in valid_datasets:
        e = f"{disease_set} not recognized. Must be one of {valid_datasets}"
        raise ValueError(e)
    if network not in valid_networks:
        e = f"{network} not recognized. Must be one of {valid_networks}"
        raise ValueError(e)
    data_path = Path("data")

    # Load the network:

    G, code_dict = build_graph_biogrid(data_path, cancertype=disease_set, filter_method=filter_method)

    # Load the disease set:
    disease_nodes_by_disease = get_cancer_genes(
        list(G.nodes()),
        code_dict,
        data_path,
        disease_set,
        load_train_genes
    )
    return G, code_dict, disease_nodes_by_disease



def get_graph(
    network: VALID_NETWORKS = "biogrid",
    cancertype: VALID_DATASETS= "ACC",
    filter_method: FilterGCC = FilterGCC.TRUE,
) -> nx.Graph:
    """Build network and disease data from raw files.

    Args:
    ----
    network: Name of the PPI network.
    filter_method: Use the full network or only the greatest connected component.

    Returns:
    -------
    Undirected Networkx graph of the PPI network.

    """
    if network not in valid_networks:
        e = f"{network} not recognized. Must be one of {valid_networks}"
        raise ValueError(e)
    data_path = Path("../data")
    if network == "biogrid":
        G, code_dict = build_graph_biogrid(data_path, cancertype = cancertype, filter_method=filter_method)
    elif network == "string":
        G, code_dict = build_graph_biogrid(data_path, cancertype = cancertype, filter_method=filter_method)
    elif network == "iid":
        G, code_dict = build_graph_biogrid(data_path, cancertype = cancertype, filter_method=filter_method)
    return G

if __name__ == "__main__":
    load_dataset("BRCA", "BioGRID", False)
