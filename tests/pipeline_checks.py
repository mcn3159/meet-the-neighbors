"""Assertions against context-only output from an actual extraction run."""
from contextlib import ExitStack
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd

from meetneighbors import context_only as co
from meetneighbors import embedding_pipeline as ep
from meetneighbors.main import get_parser, workflow


def check_context_output(testcase, run, real=False):
    reference = pd.read_csv(run / 'glm_embeds.tsv', sep='\t')
    out = run / 'context-output'
    with ExitStack() as stack:
        if not real:
            esm = run / 'glm_outputs' / co.ESM_FILENAME
            if not esm.exists():
                clusters = pd.read_csv(run / 'clust_res_in_neighborhoods' / 'mmseqs_clust.tsv', sep='\t')
                with esm.open('wb') as handle:
                    pickle.dump([(rep, np.full(1280, i, dtype=np.float16))
                                 for i, rep in enumerate(clusters.rep.unique())], handle)
            raw = co.load_esm_embeddings(esm)
            def embed(model, inputs, attention, rows, device):
                for i, row in enumerate(rows):
                    for position, protein in enumerate(row.proteins):
                        expected = -1 if position == row.center_index else raw[protein[1:]]
                        np.testing.assert_array_equal(inputs[i, position, :1280], expected)
                        testcase.assertEqual(float(inputs[i, position, 1280]), .5 if protein[0] == '+' else -.5)
                return np.asarray([np.full(1280, row.center_index, dtype=np.float32) for row in rows])
            stack.enter_context(patch.object(co, 'load_normalization', return_value=(np.zeros(1280), np.ones(1280))))
            stack.enter_context(patch.object(co, 'load_model'))
            stack.enter_context(patch.object(co, 'embed_centers', side_effect=embed))
        stack.enter_context(patch('sys.argv', [
            'meetneighbors', 'context_only', '--input-dir', str(run), '--out', str(out),
            '--input-mode', 'original', '--glm_bs', '2', '--device', 'cpu']))
        workflow(get_parser())
    context = pd.read_csv(out / co.OUTPUT_FILENAME, sep='\t')
    keys = ['query', 'rep', 'neighborhood_name', 'nn_center_id']
    pd.testing.assert_frame_equal(context[keys].sort_values(keys).reset_index(drop=True),
                                  reference[keys].sort_values(keys).reset_index(drop=True))
    testcase.assertFalse(context.duplicated(keys).any())
    testcase.assertTrue((context.source_row >= 1).all())
    testcase.assertTrue(np.isfinite(context[ep.embedding_columns(context)]).all().all())
    testcase.assertEqual(set(context.source_file), set(reference.glminput_tsv + '.tsv'))
    if not real:
        np.testing.assert_array_equal(context['0'], context.nn_center_id)
    return context
