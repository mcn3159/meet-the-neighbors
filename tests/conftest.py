"""Pytest configuration and fixtures for meet-the-neighbors tests."""

import pickle
from pathlib import Path

import pandas as pd
import dask.dataframe as dd
import pytest
import glob


@pytest.fixture(scope="session",params=['test_query_fasta','test_genomes']) # if changing make sure to also change in pytest_generate_tests
def test_data_dir(request):
    """Return the path to the test data directory."""
    return Path(__file__).parent / "test_data" / request.param


@pytest.fixture(scope="session")
def test_mode(test_data_dir):
    """
    Return the current test mode based on test_data_dir.
    
    Returns:
        str: Either 'test_query_fasta' or 'test_genomes'
    """
    return test_data_dir.name


@pytest.fixture(scope="session")
def mmseqs_clust_res(test_data_dir):
    """
    Load mmseqs clustering results.
    
    Returns:
        pandas.DataFrame: Clustering results with columns:
            rep, locus_tag, VF_center, gff, seq_id, locus_range, start, 
            strand, neighborhood_name, prot_gffname, query, cluster, nn_hashes
    """
    file_path = test_data_dir / "clust_res_in_neighborhoods" / "mmseqs_clust.tsv"
    df = dd.read_csv(file_path, sep="\t", dtype={'query':'object'},low_memory=False).compute().reset_index(drop=True)
    return df


@pytest.fixture(scope="session")
def og_mmseqs_clust(test_data_dir):
    """
    Load original mmseqs clustering results before neighborhood mapping.
    
    Returns:
        pandas.DataFrame: Original clustering with columns: rep, locus_tag
    """
    file_path = test_data_dir / "combined_fastas_clust_res.tsv"
    df = dd.read_csv(file_path, sep="\t", dtype={'query':'object'},low_memory=False,names=["rep", "locus_tag"]).compute().reset_index(drop=True)
    return df


@pytest.fixture(scope="session")
def nn_hash_obj(test_data_dir):
    """
    Load neighborhood hash mapping from pickle file.
    
    Returns:
        dict: Mapping of hash values to (query, rep, neighborhood_name) tuples
              Example: {'hash_0': [('VF001', 'ABC123', 'VF001!!!...')], ...}
    """
    file_path = test_data_dir / "nnhash_mapping.obj"
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)

@pytest.fixture(scope="session")
def glm_embeds_df(test_data_dir):
    """
    Load glm embedding results.
    
    Returns:
        pandas.DataFrame: Clustering results with columns:
            neighborhood_name, rep_y, embedding dimensions..., query
    """
    file_path = test_data_dir / "glm_embeds.tsv"
    
    df = dd.read_csv(file_path, sep="\t", dtype={'query':'object'},low_memory=False).compute().reset_index(drop=True)
    return df


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters for glm_input_df fixture."""
    # Only parametrize if the test uses glm_input_df
    if "glm_input_df" not in metafunc.fixturenames:
        return
    
    # Get the current test_data_dir parameter value
    # This respects the test_data_dir fixture's parametrization
    if "test_data_dir" in metafunc.fixturenames:
        # Get the directory where conftest.py is located
        conftest_dir = Path(__file__).parent
        test_data_base = conftest_dir / "test_data"
        
        # Get the test_data_dir parameter for this specific test
        test_data_dir_params = metafunc.definition.get_closest_marker("parametrize")
        
        # Collect TSV files only from the current test_data_dir parameter
        # We need to parametrize for ALL subdirs since test_data_dir is parametrized
        all_params = []
        
        for subdir in ['test_query_fasta', 'test_genomes']:
            glm_inputs_dir = test_data_base / subdir / "glm_inputs"
            
            if glm_inputs_dir.exists():
                tsv_files = list(glm_inputs_dir.glob("*.tsv"))
                # Store the file stem (without subdir since test_data_dir handles that)
                for f in tsv_files:
                    all_params.append(f.stem)
        
        if all_params:
            metafunc.parametrize("glm_input_df", all_params, indirect=True)


@pytest.fixture
def glm_input_df(request, test_data_dir):
    """
    Load GLM input dataframe, dynamically parametrized based on available TSV files.
    
    Parametrization: Runs once for each .tsv file found in glm_inputs directories
    
    Args:
        request: pytest request object with param attribute (tuple of subdir, vf_id)
        test_data_dir: Path to test data directory
    
    Returns:
        pandas.DataFrame: GLM input with columns:
            neighborhood_name, rep, vf_center_index
            where rep contains semicolon-delimited protein IDs with strand markers
    """
    vf_id = request.param
    file_path = test_data_dir / "glm_inputs" / f"{vf_id}.tsv"

    # df = pd.read_csv(
    #     file_path,
    #     sep="\t",
    #     names=["neighborhood_name", "rep", "vf_center_index"],
    #     dtype={"neighborhood_name": str, "rep": str, "vf_center_index": int},
    # )
    return file_path
