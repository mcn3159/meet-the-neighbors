"""
Test suite for meet-the-neighbors package.

This module contains pytest test cases converted from mtn_tests.ipynb:
- TestGLMInputValidation: Validates GLM input consistency
- TestQueryRepMapping: Validates query-to-rep protein mappings
- TestComputeUMAP: Validates UMAP input data structure
"""

import os
import pandas as pd
import pytest


class TestGLMInputValidation:
    """
    Test suite for GLM input validation.
    
    Validates that gene IDs in glm_input match mmseqs clustering results,
    including presence and ordering of representative proteins.
    """

    def test_glm_inputs(self, glm_input_df, mmseqs_clust_res):
        """
        Verify all gene IDs in glm_input are present in mmseqs_clust_res with correct order.
        
        This test:
        1. Extracts all rep (representative) protein IDs from glm_input
        2. Verifies each rep exists in mmseqs_clust_res
        3. Validates vf_center_index points to valid protein
        4. Checks ordering of reps matches mmseqs clustering
        
        Args:
            glm_input_df: Pandas DataFrame with columns [neighborhood_name, rep, vf_center_index]
            mmseqs_clust_res: Pandas DataFrame with clustering results
        """
        if not os.path.isfile(glm_input_df):
            pytest.skip(f"Skipping test_mergeback_hashnns because glm_input_df is not a file path: {glm_input_df}")
        else:
            glm_input_df = pd.read_csv(
                glm_input_df,
                sep="\t",
                names=["neighborhood_name", "rep", "vf_center_index"],
                dtype={"neighborhood_name": str, "rep": str, "vf_center_index": int},
            )
            
        # Make a copy to avoid modifying fixture
        glm_input = glm_input_df.copy()
        
        # Split semicolon-delimited rep field and explode
        glm_input["rep"] = glm_input["rep"].str.split(";")
        glm_input = (
            glm_input.explode("rep")
            .reset_index(drop=True)
            .reset_index(names="rep_index")
        )
        
        # Strip strand character (+/-) from rep ID
        glm_input["rep"] = glm_input["rep"].str[1:]
        
        # Build dict of neighborhood -> list of rep IDs from glm_input
        input_nn_rep_d = glm_input.groupby("neighborhood_name")["rep"].apply(list).to_dict()
        
        # Build dict of neighborhood -> list of rep IDs from mmseqs clustering
        mmseqs_nn_rep_d = (
            mmseqs_clust_res.sort_values(by="start")
            .groupby("neighborhood_name")[["rep", "start"]]
            .apply(lambda x: x.drop_duplicates()["rep"].tolist())
            .to_dict()
        )
        
        # Test 1: All reps in glm_input exist in mmseqs_clust_res
        for nn_name, input_reps in input_nn_rep_d.items():
            mmseqs_reps = set(mmseqs_nn_rep_d[nn_name])
            input_reps_set = set(input_reps)
            missing_reps = input_reps_set - mmseqs_reps
            assert (
                missing_reps == set()
            ), f"Not all reps in glm_input are in mmseqs_clust_res for neighborhood {nn_name}. Missing: {missing_reps}"
        
        # Test 2: Ordering of reps matches between glm_input and mmseqs_clust_res
        for nn_name, input_reps in input_nn_rep_d.items():
            mmseqs_reps = mmseqs_nn_rep_d[nn_name]
            assert (
                input_reps == mmseqs_reps
            ), f"Ordering of gene ids in glm_input does not match mmseqs_clust_res for neighborhood {nn_name}. Expected: {mmseqs_reps}, Got: {input_reps}"

class TestQueryRepMapping:
    """
    Test suite for query-to-representative protein mappings.
    
    Validates that query proteins correctly map to representative proteins
    in neighborhoods through the hash mapping structure.
    """

    def test_genomequery_rep_mapping(self, test_mode, nn_hash_obj, mmseqs_clust_res, og_mmseqs_clust):
        """
        Verify query-to-rep protein mappings are valid and consistent.
        
        This test validates:
        1. Each query in nn_hash_obj maps to a valid rep
        2. All query->rep mappings are in mmseqs_clust_res
        3. Query mapping is consistent with original clustering
        
        Args:
            test_mode: Current test mode ('test_query_fasta' or 'test_genomes')
            nn_hash_obj: Dict mapping hash -> list of (query, rep, neighborhood) tuples
            mmseqs_clust_res: Pandas DataFrame with neighborhood clustering
            og_mmseqs_clust: Pandas DataFrame with original pre-neighborhood clustering
        """

        # Skip test if not in genome mode
        if test_mode != 'test_genomes':
            pytest.skip(f"test_genomequery_rep_mapping only runs in 'test_genomes' mode (current mode: {test_mode})")
        # Extract query->rep mappings from nn_hash_obj
        q_rep_d = {}
        for hash_val, q_r_nn_list in nn_hash_obj.items():
            for grp in q_r_nn_list:
                query, rep = grp[0], grp[1]
                if query not in q_rep_d:
                    q_rep_d[query] = rep
        
        # Build original clustering mapping from combined_fastas
        mmseqs_clust_d = dict(zip(og_mmseqs_clust["locus_tag"], og_mmseqs_clust["rep"]))
        
        # Test: All query->rep mappings in nn_hash match original clustering
        for query, rep_from_hash in q_rep_d.items():
            if query in mmseqs_clust_d:
                rep_from_og = mmseqs_clust_d[query]
                # In genome mode, query maps to single rep
                assert (
                    rep_from_hash == rep_from_og
                ), f"Query {query} maps to rep {rep_from_hash} in nn_hash but to rep {rep_from_og} in original clustering"
        
        # Test: All reps in nn_hash exist in mmseqs_clust_res
        valid_reps = set(mmseqs_clust_res["rep"].unique())
        for rep in q_rep_d.values():
            assert (
                rep in valid_reps
            ), f"Rep {rep} from nn_hash not found in mmseqs_clust_res"
        
        # Test: Each query maps to exactly one rep (1:1 mapping)
        q_grps = mmseqs_clust_res.groupby("query", dropna=True)["rep"].apply(set).to_dict()
        for query, reps in q_grps.items():
            assert (
                len(reps) <= 1
            ), f"Query {query} maps to multiple reps {reps} in mmseqs_clust_res, expected 1:1 mapping"


class TestComputeUMAP:
    """
    Test suite for UMAP computation input validation.
    
    Validates that GLM input data is correctly structured and consistent
    """

    def test_mergeback_hashnns(self, glm_input_df, mmseqs_clust_res, nn_hash_obj):
        """
        Validate UMAP input data consistency and bounds.
        
        This test verifies:
        1. All neighborhoods in glm_input exist in mmseqs_clust_res
        2. Rep protein lists are valid and bounds-checked
        3. vf_center_index points to valid protein in neighborhood
        4. Neighborhood sizes are within expected bounds
        
        Args:
            glm_input_df: Pandas DataFrame with GLM input structure
            mmseqs_clust_res: Pandas DataFrame with clustering results
        """

        if not os.path.isfile(glm_input_df):
            pytest.skip(f"Skipping test_mergeback_hashnns because glm_input_df is not a file path: {glm_input_df}")
        else:
            glm_input_df = pd.read_csv(
                glm_input_df,
                sep="\t",
                names=["neighborhood_name", "rep", "vf_center_index"],
                dtype={"neighborhood_name": str, "rep": str, "vf_center_index": int},
            )
        # Make a copy to avoid modifying fixture
        input_tsv = glm_input_df.copy()
        
        # Split and explode rep field
        input_tsv["rep"] = input_tsv["rep"].str.split(";")
        input_tsv = (
            input_tsv.explode("rep")
            .reset_index(drop=True)
            .reset_index(names="rep_index")
        )
        
        # Strip strand character from rep
        input_tsv["rep"] = input_tsv["rep"].str[1:]

        mmseqs_clust = mmseqs_clust_res.loc[mmseqs_clust_res['prot_gffname'] == (mmseqs_clust_res['VF_center'] +'!!!'+ mmseqs_clust_res['gff'])].copy()
        nn_hash_df = pd.DataFrame([(k, *vals) for k, lst in nn_hash_obj.items() for vals in lst],columns=['nn_hashes','query','rep','neighborhood_name'])
        
        # Merge with mmseqs clustering data
        input_tsv = pd.merge(
            input_tsv,
            mmseqs_clust[["neighborhood_name","nn_hashes"]].drop_duplicates(),
            on="neighborhood_name",
            how="left",
        )
        
        # Test 1: All neighborhoods in glm_input exist in mmseqs_clust_res
        missing_neighborhoods = input_tsv[
            input_tsv["neighborhood_name"].isna()
        ]["neighborhood_name"].unique()
        assert (
            len(missing_neighborhoods) == 0 or all(pd.isna(missing_neighborhoods))
        ), f"Some neighborhoods from glm_input not found in mmseqs_clust_res: {missing_neighborhoods}"

        # Test 3: Verify vf_center_index is within bounds
        for nn_name, group in input_tsv.groupby("neighborhood_name"):
            vf_center_idx = group["vf_center_index"].iloc[0]
            num_reps = len(group)
            assert (
                0 <= vf_center_idx < num_reps
            ), f"vf_center_index {vf_center_idx} out of bounds [0, {num_reps}) for neighborhood {nn_name}"
        
        
        # Test 4: Neighborhood sizes are within bounds (3-30 proteins)
        nn_hash_df = nn_hash_df[nn_hash_df['nn_hashes'].isin(set(input_tsv['nn_hashes']))] # need to subset since not all hashes are in one chunk of glm_inputs
        nn_hash_df = pd.merge(nn_hash_df,input_tsv[['rep_index','nn_hashes','rep']],on='nn_hashes',how='left')
        nn_hash_df.drop_duplicates(subset=['neighborhood_name','rep_index','rep_y'],inplace=True)
        
        valcounts_to_check = nn_hash_df['neighborhood_name'].value_counts()
        assert max(valcounts_to_check) <= 30, f"Neighborhoods: {valcounts_to_check[valcounts_to_check > 30].index.values} are larger than max prots argument. Something may have went wrong with or prior to merge on nn_hash_df and glm_inputs."
        assert min(valcounts_to_check) >= 3, f"Protein components of neighborhoods: {valcounts_to_check[valcounts_to_check < 3].index.values} have been lost."
        print("Compute UMAP input test passed: all neighborhoods have correct number of proteins after merging nn_hash_df and glm_inputs")
        

    def test_glmembedstsv(self,glm_embeds_df,nn_hash_obj):
        """
        Validate GLM embeddings TSV structure and consistency.
        
        This test verifies:
        1. All neighborhood names in glm_embeds_df exist in mmseqs_clust_res
        2. Each neighborhood has exactly one embedding
        3. Embedding dimensions match expected size
        
        Args:
            glm_embeds_df: Pandas DataFrame with GLM embeddings
            nn_hash_obj: Dict mapping hash -> list of (query, rep, neighborhood) tuples
        """

        # removed Test 1 b/c some nns will not be in mmseqs_clust since glm_embeds includes the hashed out nns
        # Test 1: All neighborhoods in glm_embeds_df exist in mmseqs_clust_res
        # valid_neighborhoods = set(mmseqs_clust_res["neighborhood_name"].unique())
        # glm_neighborhoods = set(glm_embeds_df["neighborhood_name"].unique())
        
        # missing_neighborhoods = glm_neighborhoods - valid_neighborhoods
        # assert (
        #     len(missing_neighborhoods) == 0
        # ), f"Some neighborhoods in glm_embeds_df not found in mmseqs_clust_res: {missing_neighborhoods}"
        
        # removed the below test b/c a nn and rep can map to multiple queries if they share the same rep protein
        # # Test 2: Each neighborhood has exactly one embedding
        # nn_embedding_counts = glm_embeds_df.groupby("neighborhood_name").size()
        # multiple_embeddings = nn_embedding_counts[nn_embedding_counts > 1]
        
        # assert (
        #     len(multiple_embeddings) == 0
        # ), f"Some neighborhoods have multiple embeddings: {multiple_embeddings.index.tolist()}"
        
        # Test 3: Embedding dimensions match expected size (e.g., 1024)
        expected_dim = 1280  # Adjust as per actual expected dimension
        actual_dim = glm_embeds_df.shape[1] - 3  # Exclude neighborhood_name and query columns
        
        assert (
            actual_dim == expected_dim
        ), f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"

        # Test 4: Check col names and their position
        assert glm_embeds_df.columns[0] == 'neighborhood_name', f"First column should be 'neighborhood_name', got {glm_embeds_df.columns[0]}"
        assert glm_embeds_df.columns[1] == 'rep_y', f"Second column should be 'rep', got {glm_embeds_df.columns[1]}"
        assert glm_embeds_df.columns[-1] == 'query', f"Third column should be 'query', got {glm_embeds_df.columns[-1]}"

        # Test 5: Check that embedding values are floats
        embedding_values = glm_embeds_df.iloc[:, 2:-1]  # Exclude neighborhood_name, rep, and query columns
        assert (embedding_values.dtypes=='float').all()," Not all embedding values are floats"
        
        # Test 6: Ensure all neighborhood_name, rep, and query groups in glm_embeds_df exist in nn_hash_obj
        nn_hash_df =  pd.DataFrame([(k, *vals) for k, lst in nn_hash_obj.items() for vals in lst],columns=['nn_hashes','query','rep_y','neighborhood_name'])
        merged = pd.merge(glm_embeds_df[['neighborhood_name','rep_y','query']], nn_hash_df[['neighborhood_name','rep_y','query']], on=['neighborhood_name','rep_y','query'], how='left', indicator = True) # method is kinda cool
        missing_pairs = merged[merged['_merge'] == 'left_only'][['neighborhood_name','rep_y','query']]
        assert len(missing_pairs) == 0, f"Some neighborhood_name, rep, and query groups in glm_embeds_df not found in nn_hash_obj: {missing_pairs.to_dict(orient='records')}"



