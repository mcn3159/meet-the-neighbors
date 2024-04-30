# meet-the-neighbors
Medium scale identification of genomic neighbors in bacterial genomes. This program extracts neighborhoods from a set gffs and its respective protein fasta file with the same name. Extracted neighborhoods will be outputted in a fasta file. Multiple neighborhoods can be placed into the same fasta file. To determine which protein entry in the fasta file belongs to what neighborhood, each entry in the fasta file haas a header with the following naming scheme:

"Protein_id"!!!"Protein_center"!!!"gff_with_neighborhood"!!!"Contig_with_neighborhoods"!!!"Neighborhood_start_position"-"Neighborhood_end_position"!!!"Protein_start_position"!!!"strand"

### Main functionalities

- Extracts neighborhoods into a protein fasta
- Clusters similar neighborhoods found with the same intial query using DBscan (epsilon=0.15,min_samples=3)
- Plots Neighborhood results in a bubble chart with size of bubbles being 100/(# number of neighborhoods outside of cluster+1)
- Plots a scatter plot of neighborhood results
- To be Continued...

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

