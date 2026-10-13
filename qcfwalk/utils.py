"""Helper functions."""

from typing import Dict, List
import numpy as np
import pandas as pd

def scorify(S: np.ndarray, seed_list: List[int]) -> np.ndarray:
    """Sum edge-scores over the seed nodes."""
    return S[:, seed_list].sum(axis=1)

def inv_code_dict(code_dict: Dict[int, int]) -> Dict[int, int]:
    """Invert the mapping defined in code_dict."""
    return {v: k for k, v in code_dict.items()}


def process_df_ap(results_df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Aggregate simulation results to produce average precision results."""
    results_df = results_df[results_df.Iteration == iteration]
    results_df = results_df[["Model", "Ap"]]
    tdf = results_df.groupby(["Model"]).agg({"Ap": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "Ap mean", "Ap std"]]

def process_df_recall(results_df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Aggregate simulation results to produce mean recall results."""
    results_df = results_df[results_df.Iteration == iteration]
    results_df = results_df[["Model", "Recall"]]
    tdf = results_df.groupby("Model").agg({"Recall": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "Recall mean", "Recall std"]]

def process_df_mrr(results_df: pd.DataFrame, iteration: int = 300) -> pd.DataFrame:
    """Aggregate simulation results to produce mean reciprocal rank results."""
    results_df = results_df[results_df.Iteration == iteration]
    tdf = results_df
    tdf = tdf.groupby(["Model", "Disease", "Iteration"]).mean().reset_index()
    tdf["Rank"] = tdf.groupby(["Disease", "Iteration"]).rank(
        ascending=False,
        method="min",
    )["True Hits"]
    tdf["MRR"] = 1 / tdf["Rank"]
    tdf = tdf.groupby("Model").agg({"MRR": ["mean", "std"]}).reset_index()
    tdf.columns = [" ".join(col).strip() for col in tdf.columns.to_numpy()]
    return tdf[["Model", "MRR mean", "MRR std"]]