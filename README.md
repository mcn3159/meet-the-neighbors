# meet-the-neighbors
Large scale identification of genomic neighbors in bacterial genomes. This program extracts neighborhoods from a set gffs and its respective protein fasta file with the same name. Extracted neighborhoods will be outputted in a fasta file. Multiple neighborhoods can be placed into the same fasta file. To determine which protein entry in the fasta file belongs to what neighborhood, each entry in the fasta file haas a header with the following naming scheme:

"Protein_center"----"gff_with_neighborhood"----"Contig_with_neighborhoods"----"Start_position"----"End_position"

### Main functionalities

- Extracts neighborhoods into a protein fasta
- Clusters similar neighborhoods found with the same intial query using DBscan (epsilon=0.15,min_samples=3)
- Plots Neighborhood results in a bubble chart with size of bubbles being 100/(# number of neighborhoods outside of cluster+1)
- Plots a scatter plot of neighborhood results
- To be Continued...

Currently a work in prograss, needs to update environment.yml as well as adda simple way to install. 
