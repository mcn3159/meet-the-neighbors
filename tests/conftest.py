"""Self-contained artifact fixtures for query-FASTA and genome extraction."""
import logging
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from meetneighbors import embedding_pipeline as ep
from meetneighbors import glm_input_frm_neighbors as gi
from meetneighbors import plot_map_neighborhood_res as pn


@pytest.fixture(scope="session", params=["test_query_fasta", "test_genomes"])
def test_data_dir(request, tmp_path_factory):
    root = tmp_path_factory.mktemp(request.param)
    inputs = root / "glm_inputs"
    inputs.mkdir()
    records = []
    for center in ("A", "B"):
        for rep, start, strand in [("A", "2", "+"), ("B", "10", "-"), ("A", "100", "+")]:
            query = center if request.param == "test_genomes" else f"query_{center}"
            records.append(dict(neighborhood_name=f"{center}!!!g!!!ctg!!!2-200", rep=rep,
                                start=start, strand=strand, prot_gffname=f"{rep}!!!g",
                                VF_center=center, gff="g", query=query if rep == center else None))
    records.append(records[0].copy())  # redundant search row at the same occurrence
    if request.param == "test_query_fasta":
        records.append({**records[0], "query": "second_label"})
    clusters, mapping = pn.hash_neighborhoods(pd.DataFrame(records), SimpleNamespace())
    (root / "clust_res_in_neighborhoods").mkdir()
    clusters.to_csv(root / "clust_res_in_neighborhoods" / "mmseqs_clust.tsv", sep="\t", index=False)
    (root / "combined_fastas_clust_res.tsv").write_text("A\tA\nB\tB\n")
    with (root / "nnhash_mapping.obj").open("wb") as handle:
        pickle.dump(mapping, handle)
    gi.get_glm_input(query="fixture", uniq_neighborhoods_d={"fixture": set(clusters.neighborhood_name)},
                     mmseqs_clust=clusters, glm_input_dir="glm_inputs", logger=logging.getLogger(__name__),
                     args=SimpleNamespace(out=str(root) + "/"))
    rows = ep.read_neighborhoods(inputs, clusters)
    def embed(batch):
        return np.broadcast_to(np.arange(30, dtype=np.float32)[None, :, None], (len(batch), 30, 1280)).copy()
    ep.run_embedding_batches(rows, clusters, mapping, root / "glm_outputs", embed, 1, 30)
    return root


@pytest.fixture(scope="session")
def test_mode(test_data_dir):
    return "test_genomes" if test_data_dir.name.startswith("test_genomes") else "test_query_fasta"


@pytest.fixture(scope="session")
def mmseqs_clust_res(test_data_dir):
    return pd.read_csv(test_data_dir / "clust_res_in_neighborhoods" / "mmseqs_clust.tsv", sep="\t")


@pytest.fixture(scope="session")
def og_mmseqs_clust(test_data_dir):
    return pd.read_csv(test_data_dir / "combined_fastas_clust_res.tsv", sep="\t", names=["rep", "locus_tag"])


@pytest.fixture(scope="session")
def nn_hash_obj(test_data_dir):
    with (test_data_dir / "nnhash_mapping.obj").open("rb") as handle:
        return pickle.load(handle)


@pytest.fixture(scope="session")
def glm_embeds_df(test_data_dir):
    return pd.read_csv(test_data_dir / "glm_embeds.tsv", sep="\t")


@pytest.fixture
def glm_input_df(test_data_dir):
    return test_data_dir / "glm_inputs" / "fixture.tsv"
