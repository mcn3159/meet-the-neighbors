"""SLURM staging/worker tests use fake sbatch and never submit real jobs."""
import csv
import fcntl
import json
import logging
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dask
import numpy as np
import pandas as pd

from meetneighbors import embedding_pipeline as ep
from meetneighbors import neighbors_frm_mmseqs as neighbors
from meetneighbors import ring_mmseqs as mm
from meetneighbors.main import get_parser, workflow
from meetneighbors.predictvfs import neighborhood_classification as nc
from meetneighbors.predictvfs.glm import glm_embed as ge

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(os.environ.get('MTN_ARRAY_SCRIPT', str(REPO.parent / 'slurm_scripts/update_glm_embeds_array.sh')))
UPDATE = 'update_embeds_11102026'
BOOTSTRAP = ('vfs_in_genomes.tsv', 'combined_fastas_clust_res.tsv', 'combined_fastas_clust_rep.fasta')


class ArrayScriptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        data = self.root / 'data'
        self.sources = [data / 'training_set/sec_vf_refseqs_noredolp_ig2', self.root / 'phrog']
        self.sources += [data / f'training_set/uniprotkb_nonvfs_ns30k_ig10k/part_{i:03}' for i in range(1, 7)]
        query_dir = data / 'validation_set/nonvfs_virulenthunter_protocol/mmseqsdb/clu_og_nonvfsdb/nonvf_reps_nobenchmarkprots_3132026_sub_flt.fasta.split'
        queries = [data / 'vfdb_fastas/signalp_res/setb_secreted_clust/reps/vfdb_secreted_setb_DB_clu_rep.fasta',
                   data / 'training_set/phrog/clu/reps_rand50newids.fasta']
        queries += [query_dir / f'nonvf_reps_nobenchmarkprots_3132026_sub_flt.part_{i:03}.fasta' for i in range(1, 7)]
        for path in queries:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('>A\nMKKL\n')
        for path in self.sources:
            path.mkdir(parents=True)
            for name in (*BOOTSTRAP, 'glm_embeds.tsv', 'nnhash_mapping.obj'):
                (path / name).write_text('original ' + name)
        genomes = self.root / 'genomes'
        genomes.mkdir()
        database = self.root / 'genomesdb'
        database.write_text('database')
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name, capture in [('sbatch', 'submit.json'), ('meetneighbors', 'worker.json')]:
            executable = self.bin / name
            executable.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                                  f'with open({str(self.root / capture)!r}, "w") as handle:\n'
                                  '    json.dump({"argv": sys.argv[1:], "pythonpath": os.environ.get("PYTHONPATH")}, handle)\n'
                                  'print("12345")\n')
            executable.chmod(0o755)
        # The worker activates the requested environment, without requiring Conda
        # in unit tests; its Python and executable are supplied on PATH.
        (self.bin / 'python').symlink_to(sys.executable)
        conda = self.root / 'conda.sh'
        conda.write_text('conda() { [[ "$1" == activate && "$2" == mtn_colab ]]; }\n')
        self.env = {**os.environ, 'MTN_DATA_ROOT': str(data), 'MTN_PHROG_OUT': str(self.sources[1]),
                    'MTN_GENOMES': str(genomes), 'MTN_GENOMES_DB': str(database),
                    'MTN_REPO_ROOT': str(REPO), 'MTN_CONDA_SH': str(conda),
                    'MTN_JOB_ROOT': str(self.root / 'jobs'), 'PATH': str(self.bin) + os.pathsep + os.environ['PATH']}

    def run_script(self, *args, task=None, ok=True):
        env = self.env.copy()
        if task is not None:
            env['SLURM_ARRAY_TASK_ID'] = str(task)
        else:
            env.pop('SLURM_ARRAY_TASK_ID', None)
        result = subprocess.run(['bash', str(SCRIPT), *map(str, args)], env=env, text=True, capture_output=True)
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def submit(self):
        self.run_script('--submit')
        return next((self.root / 'jobs').glob('*/runs.tsv'))

    def test_shell_syntax_and_read_only_dry_run(self):
        subprocess.run(['bash', '-n', str(SCRIPT)], check=True)
        before = {str(p.relative_to(self.root)) for p in self.root.rglob('*')}
        result = self.run_script('--dry-run')
        self.assertIn('8 tasks; SLURM array 0-7', result.stdout)
        self.assertEqual(result.stdout.count('  Command:'), 8)
        self.assertIn('--threads 7 --mem 110 --glm_bs 5000', result.stdout)
        self.assertEqual(before, {str(p.relative_to(self.root)) for p in self.root.rglob('*')})

    def test_submit_freezes_assignments_and_workers_preserve_originals(self):
        manifest = self.submit()
        submitted = json.loads((self.root / 'submit.json').read_text())['argv']
        self.assertIn('--array=0-7', submitted)
        self.assertIn('--worker', submitted)
        self.assertEqual(submitted[-1], str(manifest))
        with manifest.open() as handle:
            rows = list(csv.DictReader(handle, delimiter='\t'))
        self.assertEqual([row['source'] for row in rows], list(map(str, self.sources)))
        (self.sources[-1].parent / 'part_000').mkdir()  # must not renumber worker tasks
        originals = {path: ep.file_digest(path) for path in self.sources[2].iterdir()}
        self.run_script('--worker', manifest, task=2)
        output = self.sources[2] / UPDATE
        for name in BOOTSTRAP:
            self.assertTrue((output / name).is_symlink())
            self.assertTrue((output / name).samefile(self.sources[2] / name))
        self.assertFalse((output / 'glm_embeds.tsv').exists())
        called = json.loads((self.root / 'worker.json').read_text())
        self.assertEqual(called['argv'][called['argv'].index('--out') + 1], str(output))
        self.assertIn('--resume', called['argv'])
        self.assertTrue(called['pythonpath'].startswith(str(REPO / 'src')))
        self.run_script('--worker', manifest, task=2)  # matching links remain usable
        self.assertEqual(originals, {path: ep.file_digest(path) for path in originals})

    def test_missing_input_fails_before_submission(self):
        (self.sources[4] / BOOTSTRAP[0]).unlink()
        result = self.run_script('--submit', ok=False)
        self.assertIn('Missing required file', result.stderr)
        self.assertFalse((self.root / 'jobs').exists())
        self.assertFalse((self.root / 'submit.json').exists())

    def test_conflicting_bootstrap_and_output_links_are_rejected(self):
        output = self.sources[0] / UPDATE
        output.mkdir()
        link = output / BOOTSTRAP[0]
        link.write_text('do not overwrite')
        self.assertIn('Conflicting bootstrap', self.run_script('--dry-run', ok=False).stderr)
        self.assertEqual(link.read_text(), 'do not overwrite')
        link.unlink()
        link.symlink_to(self.sources[1] / BOOTSTRAP[0])
        self.assertIn('Conflicting bootstrap', self.run_script('--dry-run', ok=False).stderr)
        link.unlink()
        link.symlink_to(self.sources[0] / BOOTSTRAP[0])
        (output / 'glm_outputs').symlink_to(self.sources[0])
        self.assertIn('Unexpected output symlink', self.run_script('--dry-run', ok=False).stderr)

    def test_invalid_indices_and_lock_contention(self):
        manifest = self.submit()
        for task in (None, -1, 'bad', 8):
            self.run_script('--worker', manifest, task=task, ok=False)
        output = self.sources[0] / UPDATE
        self.assertFalse(output.exists())
        output.mkdir()
        with (output / '.update_glm.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_script('--worker', manifest, task=0, ok=False)
            self.assertIn('Another update worker', result.stderr)
        self.assertFalse((self.root / 'worker.json').exists())


class ThreeFileResumeTests(unittest.TestCase):
    def test_bootstrap_rebuilds_metadata_without_search_extraction_or_clustering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'original'
            source.mkdir()
            output = source / UPDATE
            output.mkdir()
            query = root / 'query.fasta'
            query.write_text('>Q_A\nMKKLALAL\n>Q_B\nMTTGGVVA\n')
            genomes = root / 'genomes'
            genomes.mkdir()  # No GFFs: extraction must not run.
            raw = []
            for center in ('A', 'B'):
                for rep, start, strand in [('A', 2, '+'), ('B', 10, '-'), ('A', 100, '+')]:
                    raw.append(f'{rep}\t{rep}!!!{center}!!!g!!!ctg!!!2-200!!!{start}!!!{strand}\n')
            (source / 'combined_fastas_clust_res.tsv').write_text(''.join(raw))
            (source / 'combined_fastas_clust_rep.fasta').write_text('>A\nMKKLALAL\n>B\nMTTGGVVA\n')
            pd.DataFrame({'query': ['Q_A', 'Q_B'], 'target': ['A', 'B'],
                          'tset': ['g.faa', 'g.faa'], 'tsetid': [0, 0]}).to_csv(
                              source / 'vfs_in_genomes.tsv', sep='\t', index=False)
            (source / 'glm_embeds.tsv').write_text('old embeddings remain here')
            (source / 'nnhash_mapping.obj').write_bytes(b'old mapping must not be loaded')
            (source / 'glm_outputs').mkdir()
            (source / 'glm_outputs' / 'legacy.pkl').write_bytes(b'old vectors')
            originals = {p: ep.file_digest(p) for p in source.rglob('*') if p.is_file()}
            for name in BOOTSTRAP:
                (output / name).symlink_to(source / name)
            checkpoint = root / 'checkpoint'
            checkpoint.write_bytes(b'fixture checkpoint identity')
            norms = {'norm.pkl': {'mean': np.zeros(1280), 'std': np.ones(1280)}}
            argv = ['meetneighbors', 'extract_neighbors', '--query_fasta', str(query),
                    '--genomes', str(genomes), '--out', str(output), '--threads', '1',
                    '--mem', '110', '--glm_bs', '2', '--max_prots', '5',
                    '--cluster', '--memory_optimize', '--resume']
            def embed(reps, model, rows, mean, std, maximum, layer):
                self.assertEqual(layer, 3)
                nc.prepare_batch(rows, reps, mean, std, maximum)
                return np.broadcast_to(np.arange(maximum)[None, :, None], (len(rows), maximum, 1280)).copy()
            with patch('sys.argv', argv), dask.config.set(scheduler='synchronous'), \
                 patch('meetneighbors.main.get_logger', return_value=logging.getLogger('bootstrap-test')), \
                 patch.object(mm, 'mmseqs_createdb', side_effect=AssertionError('createdb')), \
                 patch.object(mm, 'mmseqs_search', side_effect=AssertionError('search')), \
                 patch.object(mm, 'mmseqs_cluster', side_effect=AssertionError('clustering')), \
                 patch.object(neighbors, 'get_neigborhood', side_effect=AssertionError('extraction')), \
                 patch.object(nc, 'get_plm_embeds', return_value={p: [np.zeros(1280), p] for p in 'AB'}) as plm, \
                 patch.object(ge, 'load_model', return_value=object()) as model, \
                 patch.object(nc, 'embed_sequences', side_effect=embed) as inference, \
                 patch('meetneighbors.predictvfs.loader.load_pickle', return_value=norms), \
                 patch('meetneighbors.predictvfs.loader.get_glm_model_path', return_value=checkpoint):
                workflow(get_parser())
                plm.assert_called_once()
                model.assert_called_once()
                inference.assert_called_once()
                frame = pd.read_csv(output / 'glm_embeds.tsv', sep='\t')
                self.assertEqual(set(zip(frame['query'], frame.nn_center_id)), {('Q_A', 0), ('Q_A', 2), ('Q_B', 1)})
                np.testing.assert_array_equal(frame['0'], frame.nn_center_id)
                with (output / 'nnhash_mapping.obj').open('rb') as handle:
                    mapping = pickle.load(handle)
                self.assertTrue(all(len(entry) == 4 for entries in mapping.values() for entry in entries))
                before = ep.file_digest(output / 'glm_embeds.tsv')
                plm.side_effect = model.side_effect = inference.side_effect = AssertionError('recomputed completed embeddings')
                workflow(get_parser())
                self.assertEqual(ep.file_digest(output / 'glm_embeds.tsv'), before)
            self.assertEqual(originals, {p: ep.file_digest(p) for p in originals})


if __name__ == '__main__':
    unittest.main()
