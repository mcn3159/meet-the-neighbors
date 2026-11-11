# meet-the-neighbors
Medium scale identification of genomic neighbors in bacterial genomes. This program extracts neighborhoods from a set gffs and its respective protein fasta file with the same name. Extracted neighborhoods will be outputted in a fasta file, with the protein identifier denoting the neighborhood name and protein ID. Multiple neighborhoods can be placed into the same fasta file. 

To determine which protein entry in the fasta file belongs to what neighborhood, each entry in the fasta file haas a header with the following naming scheme:

"Protein_id"!!!"Protein_center"!!!"gff_with_neighborhood"!!!"Contig_with_neighborhoods"!!!"Neighborhood_start_position"!!!"Neighborhood_end_position"!!!"Protein_start_position"!!!"strand"

### Main functionalities

- Extracts neighborhoods into a protein fasta
- Clusters similar neighborhoods found with the same intial query using DBscan (epsilon=0.15,min_samples=3)
- Plots Neighborhood results in a bubble chart with size of bubbles being 100/(# number of neighborhoods outside of cluster+1)
- Outputs neighborhoods in compatible format for a genomic language model (https://github.com/y-hwang/gLM)
- To be continued...

### Running Program
```
git clone
```

Activate the conda environment
```
conda env create -f environment.yml
conda activate meet-the-neighbors
```

Basic run command 
```
python main.py extract_neighbors --query_fasta dir/to/queries.fasta --genomes dir/to/genomes_to_search --out dir/to/out_directory --plot
```
Each genome to extract the neighborhoods from must occur in a gff and protein fasta formats, with the same file name followed by an underscore _genomic or _protein respectively.
For example:

```
cd genomes/

GCA_000307975.2_ASM30797v2_genomic.gff
GCA_000307975.2_ASM30797v2_protein.faa
...
```

### Interactive neighborhood visualization

`viz_neighborhoods` turns an existing `extract_neighbors` run into a standalone
`neighborhoods.html` report. Open the file in a browser; it works offline. Gene
arrows preserve nucleotide lengths and spacing, align at the center gene, and
reverse neighborhoods whose center is on the minus strand. Overlapping genes
occupy separate lanes. Hover or keyboard-focus a gene to see its original GFF
`product`, protein ID, cluster representative, coordinates, and strand. Matching
colors identify clusters present in two or more displayed neighborhoods;
hovering highlights all members of the same cluster. A dark outline marks each
center. The report includes a cluster legend and horizontal zoom/reset controls.

Supply the extraction directory plus the original GFF directory. Use exact
`neighborhood_name` values from the run's tables or encoded FASTA headers, quoted
as shown (the names contain `!!!`):

```bash
# One neighborhood (supply more names after --neighborhood-names for a list).
meetneighbors viz_neighborhoods \
  --input-dir /path/to/extraction --genomes /path/to/genomes \
  --out /path/to/viz_one \
  --neighborhood-names 'proteinA!!!genome1!!!contig1!!!100-10000'

# Ten random neighborhoods; --seed makes the selection reproducible.
meetneighbors viz_neighborhoods \
  --input-dir /path/to/extraction --genomes /path/to/genomes \
  --out /path/to/viz_random --n-random 10 --seed 42

# Explicit neighborhoods, followed by five additional random neighborhoods.
meetneighbors viz_neighborhoods \
  --input-dir /path/to/extraction --genomes /path/to/genomes \
  --out /path/to/viz_mixed --n-random 5 --seed 42 \
  --neighborhood-names 'proteinA!!!genome1!!!contig1!!!100-10000' \
                       'proteinB!!!genome2!!!contig2!!!200-11000'
```

Explicit selections retain their order and are deduplicated. Random sampling is
without replacement and excludes explicit selections. The default is ten random
neighborhoods when no names are supplied, or zero additional neighborhoods when
names are supplied. Requests exceeding the available random pool include all
remaining neighborhoods with a warning. Names not found in the input directory
are skipped with a warning; matching names retain their order and are processed
normally. If none of the requested names are found, the command errors even if
`--n-random` is positive. With no names supplied, random-only selection still
works normally. Reclustering uses only proteins from the final selection (matching
explicit names plus any requested random additions). Empty selections are errors.
Existing `neighborhoods.html` files are never overwritten.

Instead of `--genomes`, use `--genome_tsv /path/to/sources.tsv` (also spelled
`--genome-tsv`). This is a **headerless** TSV with columns `genome name`, `protein
FASTA path`, and `GFF path`; relative paths resolve against the TSV's directory.
Genome names must match those in `neighborhood_name`. Directory lookup supports
`<genome>.gff` / `<genome>_genomic.gff` and `<genome>.faa`, `<genome>.fasta`, or
`<genome>_protein.faa`. Ambiguous filenames require an explicit TSV mapping.
Protein FASTAs are needed only for reclustering.

The viewer combines original neighborhood membership from `neighborhoods_db.tsv`,
encoded `combined_fasta_partition*` headers (or `combined.fasta`), and saved
cluster tables. This retains neighborhoods omitted from the reduced table by
memory optimization when their original membership artifacts are still present.
Original GFFs supply gene ends and products, which existing extraction tables do
not preserve. Missing products are labeled “No product annotation”; missing or
ambiguous CDS matches are errors. Displayed genes are only those actually saved
by extraction, including its original strand and neighborhood filtering.

Saved MMseqs2 assignments default to
`clust_res_in_neighborhoods/mmseqs_clust.tsv`. When that table exists, assignments
from `combined_fastas_clust_res.tsv` are ignored, including for genes missing from
the neighborhood table. The original TSV remains available for recovering
neighborhood membership and supplies cluster assignments only when the neighborhood
table is absent. Cluster assignments from the two files are never merged.
Conflicts affecting selected neighborhoods produce an error identifying the file
and conflicting row, with guidance to use `--recluster`. Conflicts in assignments
not used by the selection are skipped with a warning, so unrelated neighborhoods
cannot block a visualization. No conflicting representative is chosen arbitrarily.
Without evidence of
genuine clustering, synthetic self-assignments from unclustered extraction runs
are shown as unclustered. Missing assignments remain gray. A bare protein ID
that cannot distinguish genomic occurrences is rejected; reclustering resolves
that ambiguity using unique occurrence IDs.

To recompute protein clusters for **selected neighborhoods only**, install
MMseqs2 and make `mmseqs` available on `PATH`, then run:

```bash
meetneighbors viz_neighborhoods \
  --input-dir /path/to/extraction --genomes /path/to/genomes \
  --out /path/to/viz_reclustered --n-random 10 --seed 42 \
  --recluster --cluster-dir /path/to/new_clusters \
  --seq-id 0.5 --cov 0.8 --cov-mode 0 --sensitivity 7 \
  --cluster-mode 0 --threads 4 --mem 8
```

Identity and coverage are fractions, defaulting to 0.90 each. Coverage mode
defaults to 0, threads to 4, and the MMseqs2 split-memory limit to 8 GB. Sensitivity
and cluster mode use MMseqs2 defaults unless supplied. Clustering settings apply
only with `--recluster`. The new clustering directory defaults to `<out>/recluster`
and must not already exist. It contains `selected_proteins.fasta`,
`protein_mapping.tsv` (including each neighborhood occurrence), `clusters.tsv`,
`representatives.fasta`, MMseqs2 databases, `run.json` (settings, commands, version,
and completion status), and `mmseqs.log`. Shared genomic occurrences are clustered
once, then mapped back to every selected neighborhood. Failures retain diagnostic
files and do not produce a completed HTML report. Extraction files are unchanged.

The Python API requires no plotting or machine-learning imports:

```python
from meetneighbors.viz_neighborhoods import viz_neighborhoods

html_path = viz_neighborhoods(
    input_dir="/path/to/extraction",
    out="/path/to/viz_python",
    neighborhood_names=["proteinA!!!genome1!!!contig1!!!100-10000"],
    n_random=5,
    seed=42,
    genomes="/path/to/genomes",
)
# neighborhood_names also accepts a single string, or None for random-only.
# Optional: recluster=True, cluster_dir="/path/to/new_clusters", seq_id=0.5, cov=0.8
print(html_path)
```

Run the self-contained visualization tests without the ML dependencies:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_viz_neighborhoods.py -v
```

The real MMseqs2 smoke test runs when `mmseqs` is installed; other clustering
tests use a mock executable interface.

### Batched extraction and VF embeddings

`extract_neighbors` and `predictvf` compute layer-3 GLM hidden states directly,
without prediction heads, next-gene probabilities, losses, or PCA labels.
Only the encoder layers needed for layer 3 are loaded and executed; supported
PyTorch versions memory-map the checkpoint to avoid loading unused weights.
`--glm_bs` bounds the number of neighborhoods per GLM inference batch. Every
protein position is embedded, then VF centers are selected by their occurrence
index, including centers restored from deduplicated neighborhoods.

Each `glm_outputs/batch_000000.glm_embeds.tsv` (and subsequent numbered batch)
contains restored VF-center rows. `glm_embeds.tsv` combines those batches.
Both have columns `glminput_tsv`, `query`, `neighborhood_name`, `rep`,
`nn_center_id`, then embedding dimensions `0` through `1279`. `nn_center_id` is
zero-based; `glminput_tsv` is the source TSV stem after input chunking.
For un-hashed runs, every matching VF-center occurrence in `mmseqs_clust.tsv`
gets a row, including duplicated gene names. The input TSV still has one row
per neighborhood; its center index is a reference, not a restriction to one
copy. Each neighborhood is embedded once. Prediction outputs retain `rep` and
`nn_center_id` so copies with identical query and neighborhood names remain
distinguishable. Existing indexed hash mappings likewise retain each specified
center position; extraction and hash generation are unchanged by this handling.

New `nnhash_mapping.obj` entries are `(query, rep, neighborhood_name, center_index)`.
Hashes use the historical ordered representative tuple (without strands), with
one sampled neighborhood retained per hash. Center indices count distinct
numeric start positions, so redundant search hits do not shift duplicated
genes' positions. All matching center occurrences are retained in the mapping.
`--resume` verifies batch
checksums and input/model configuration before reuse; changed configurations
require a run without `--resume`. Batch manifests now use version 2; cached
batches from before duplicate-center support must be regenerated without
`--resume`. Older three-field hash mappings emit a warning
and use the historical `compute_umap.py` occurrence-selection behavior, while
newly computed embeddings still use layer 3.

Genome-only runs with `--mem >= 16` link the combined protein FASTA into
`glm_inputs`. Both unclustered `combined.fasta` and clustered
`combined_fastas_clust_rep.fasta` sources work with relative output paths.
Resuming an older run repairs a dangling genome FASTA link before checking
embedding inputs.

Self-contained regression tests (no model downloads):

```bash
PYTHONPATH=src:tests python -m pytest tests -q
```

The real-checkpoint CPU smoke test is opt-in via `MEETNEIGHBORS_TEST_REAL_GLM=1`;
it loads GLM only, using synthetic protein embeddings. The normal tests use
small models and deterministic embeddings for extraction integration tests.
For end-to-end tests with cached ESM/GLM/classifier weights and MMseqs installed,
set `MEETNEIGHBORS_TEST_REAL_GENOME=1`. This tests a small real NR2335 slice
containing a repeated gene, through genome-only and query-FASTA extraction,
prediction, resume, and context-only inference. Run serially with a process
memory monitor when testing on a memory-constrained machine.

### Context-only GLM embeddings

Context-only and extraction share the GLM checkpoint loader and hidden-state
inference code. Context-only loads through layer 9 and masks the selected
occurrence before normalization; extraction uses unmasked layer-3 embeddings.

Use the inputs prepared by the `context_only.ipynb` loop:

```bash
meetneighbors context_only \
  --input-dir /path/to/masked_embeds \
  --neighborhood-run /path/to/original_mtn_run \
  --out /path/to/context_only_results \
  --glm_bs 100 \
  --device auto
```

The input directory must contain `glm_inputs/*.tsv` and
`glm_outputs/all_glm_input_prots_reps.esm.embs.pkl`. Each headerless TSV row is
`neighborhood_name<TAB>semicolon-separated stranded protein IDs<TAB>zero-based center index`.
Protein IDs start with `+` or `-`. The default `--input-mode masked` expects the
notebook's terminal `_masked` suffix on each center ID. To use unmodified
meet-the-neighbors outputs directly, pass `--input-mode original` and point
`--input-dir` at the original run directory. No ESM recomputation is needed.

For each requested representative occurrence, the command replaces its **raw 1,280-dimensional
ESM vector with -1 before normalization**. Normalization therefore changes those
values; GLM does not receive literal -1 features. The center's strand and position
are retained, and neighboring occurrences of the same protein keep their own ESM
vectors. For memory-optimized runs, protein positions are reconstructed from the
input TSVs and labels are restored through `nnhash_mapping.obj`, matching
the saved center index for new four-field hash entries. For legacy three-field
entries, the first matching occurrence is selected, matching `compute_umap.py`'s
query/neighborhood deduplication. The retained neighborhood's VF-center index is
not transferred to restored queries. For ordinary runs without hash mapping,
`nn_center_id` in the reference embedding table selects each occurrence. Legacy
tables without that field retain the input TSV's center when applicable;
other targets require a unique position.
In masked mode, original ESM vectors from `--neighborhood-run` are
used when available, restoring the old center before masking a different target.

The command uses the existing GLM weights and normalization factors. Set
`MEETNEIGHBORS_GLM_BIN=/path/to/glm.bin` if the checkpoint is not in the repository's
`src/meetneighbors/predictvfs/glm/model/` directory. `--device auto` selects CUDA
when available, otherwise CPU; `--device cpu` and `--device cuda` explicitly choose
the device. CUDA uses mixed precision. CPU uses float32 inference. Input staging
uses the existing batcher's float16 precision. Neighborhoods must contain 1–30 proteins.

The output is `context_only_layer9.tsv`, with `source_file`, one-based `source_row`,
`neighborhood_name`, `rep`, `query`, zero-based `nn_center_id`, and embedding
columns `0`–`1279`. Every distinct `(neighborhood_name, rep, query, nn_center_id)`
in the original `glm_embeds.tsv`
is included, accepting its legacy `rep_y` column as `rep`. Reference embedding
values are not used. For hashed runs the independently reconstructed metadata
must exactly match the table: missing, extra, or mismatched query labels stop the
run before inference. Use inputs, hash mapping, and reference from the same
complete run. If the table is absent, labels are reconstructed using the
existing GLM assembly rules: hash mappings for memory-optimized runs, or input
filename query labels and explicit centers for ordinary runs.

`--neighborhood-run` is optional when the original run can be found in the input
directory or an ancestor. Its `nnhash_mapping.obj` restores omitted neighborhoods.
The source columns always identify the retained input used for inference, even
when the output neighborhood name was restored through that mapping. Output
follows reference metadata order (first occurrence of each occurrence-qualified label), or a
deterministic reconstructed order when no reference exists. Missing mappings
fail before model inference instead of silently omitting labels.

Reference validation errors identify the one-based file line (including the
header), missing fields, one-based metadata column numbers, escaped metadata
values, and malformed row widths, without printing embedding values. Empty query
fields can originate in the legacy assembly's query merge/explode. For these
errors, up to four nonempty query candidates from the loaded hash mapping are
shown for the same neighborhood and representative. Candidates are diagnostic
only: repair the original query mapping and regenerate consistent metadata;
the command does not guess labels or silently drop unlabeled rows.

Identical normalized masked inputs, attention masks, and extraction positions
share one inference result, even across different queries or masked protein IDs.
Neighborhood hashes alone are not used for embedding reuse because they omit
strand information. Mapping and vector caches use temporary SQLite files under
the output directory, removed on completion or failure. Source occurrences are
indexed and reused by `(hash, rep)` without building the legacy hash-group
Cartesian product. Allow disk space for the
cached float32 vectors (5,120 bytes per unique context, plus indexing overhead).
The mapping pickle is loaded once; reference TSVs are streamed without retaining
their embedding matrix. Layer 9 means hidden-state index 9, where index 0 is the
embedding-stage output. No classifier is run.

The source files remain unchanged. Existing output tables are rejected. Invalid
center indices, strand prefixes, missing neighbor embeddings, invalid ESM vectors,
and oversized neighborhoods are reported before model inference. Paths do not
require trailing slashes. Original center IDs are restored by removing just one
terminal `_masked` suffix in masked mode. The unused notebook `allmaskedembeds`
branch is not part of this implementation.

Small CPU regression tests include the two reported duplicate-occurrence cases
and compare labels and protein positions against the actual `compute_umap.py`
assembler using position-labelled dummy vectors:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_context_only*.py'
```

These tests do not load production mapping pickles or the full GLM checkpoint.
Independent small-model comparisons allow `rtol=1e-3`, `atol=5e-3`, and require
cosine similarity of at least `0.99999` for equal contexts.
