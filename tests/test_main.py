"""Validate current saved artifacts in both query-FASTA and genome modes."""
import numpy as np
import pandas as pd

from meetneighbors import embedding_pipeline as ep
from meetneighbors.neighborhood_order import ordered_occurrences


class TestGLMInputValidation:
    def test_glm_inputs(self, glm_input_df, mmseqs_clust_res):
        rows = ep.read_neighborhoods(glm_input_df.parent, mmseqs_clust_res)
        assert {row.neighborhood_name for row in rows} == set(mmseqs_clust_res.neighborhood_name)
        for row in rows:
            ordered = ordered_occurrences(mmseqs_clust_res.loc[
                mmseqs_clust_res.neighborhood_name.eq(row.neighborhood_name)])
            assert row.proteins == tuple(ordered.strand + ordered.rep)
            assert 0 <= row.center_index < len(row.proteins)
            assert row.center_id == ordered.rep.iloc[row.center_index]


class TestQueryRepMapping:
    def test_query_rep_mapping(self, test_mode, nn_hash_obj, mmseqs_clust_res, og_mmseqs_clust):
        original = dict(zip(og_mmseqs_clust.locus_tag, og_mmseqs_clust.rep))
        valid_reps = set(mmseqs_clust_res.rep)
        for entries in nn_hash_obj.values():
            for query, rep, name, position in entries:
                assert query and name
                assert rep in valid_reps
                assert isinstance(position, int) and position >= 0
                if test_mode == "test_genomes":
                    assert query in original
                    assert original[query] == rep
        for reps in mmseqs_clust_res.groupby("query").rep.apply(set):
            assert len(reps) == 1


class TestComputeUMAP:
    def test_mergeback_hashnns(self, glm_input_df, mmseqs_clust_res, nn_hash_obj):
        rows = ep.read_neighborhoods(glm_input_df.parent, mmseqs_clust_res)
        targets = ep.center_targets(rows, mmseqs_clust_res, nn_hash_obj)
        expected = {tuple(entry) for entries in nn_hash_obj.values() for entry in entries}
        actual = set()
        for source, position, metadata in targets:
            stem, query, name, rep, center = metadata
            assert stem == glm_input_df.stem
            assert 3 <= len(rows[source].proteins) <= 30
            assert 0 <= position < len(rows[source].proteins)
            assert position == center
            assert rows[source].proteins[position][1:] == rep
            actual.add((query, rep, name, center))
        assert actual == expected
        assert len(targets) == len(expected)

    def test_glmembedstsv(self, glm_embeds_df, nn_hash_obj):
        frame = glm_embeds_df
        dimensions = ep.embedding_columns(frame)
        assert list(frame.columns) == ep.METADATA + [str(i) for i in ep.DIMENSIONS]
        assert len(dimensions) == 1280
        assert all(pd.api.types.is_float_dtype(dtype) for dtype in frame[dimensions].dtypes)
        assert np.isfinite(frame[dimensions].to_numpy()).all()
        keys = ["query", "rep", "neighborhood_name", "nn_center_id"]
        expected = {tuple(entry) for entries in nn_hash_obj.values() for entry in entries}
        assert set(frame[keys].itertuples(index=False, name=None)) == expected
        assert not frame.duplicated(keys).any()
        # Position-encoded fixtures prove that repeated reps select their own vector.
        np.testing.assert_array_equal(frame["0"], frame.nn_center_id)
