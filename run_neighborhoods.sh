# $1 input fasta (VFDB fasta)
# $2 directory for genomes, should contain fastas with match gff names
# $3 output directoy
# $4 threads
# $5 2/3 ram limit

#Logic: 1)Search for input (vfdb) proteins in the fastas of your genomes of interest. 
# 2) From those search hits, pull their neighborhoods from the gffs
# 3) Cluster all the proteins within all neighborhoods, pivot and organize new mini dfs with unique neighborhood id as index, prot cluster as col, for each query
mmseqs createdb $1 queryDB

faa_dir="$2/*.faa" #make sure there's no slash after the folder name. First param should be full path to folder with fastas
mmseqs createdb $faa_dir ab_fastasDB

mmseqs search queryDB ab_fastasDB vf_in_abDB tmp_search --min-seq-id 0.9 --cov-mode 0 -c 0.9 --split-memory-limit $5 --threads $4 --start-sens 1 --sens-steps 3 -s 7
mmseqs convertalis queryDB ab_fastasDB vf_in_abDB vfs_inOrgs.tsv --format-output "query,target,evalue,pident,qcov,fident,alnlen,qheader,theader,tset,tsetid" 

python neighbors_frm_mmseqs.py  --genomes "$2/" --out $3 --threads $4 --frm_vfdb

mmseqs createdb combined_fasta_partition* combined_fastas_db
mmseqs linclust combined_fastas_db combined_fastas_clust --cov-mode 0 -c 0.90 --min-seq-id 0.90 --similarity-type 2 --split-memory-limit $5 --threads $4 tmp_clust
mmseqs createtsv combined_fastas_db combined_fastas_db combined_fastas_clust combined_fastas_clust_res.tsv

python plot_map_neighborhood_res.py --mmseqs_tsv neighbors_all_ids_clust_res.tsv --vfdb_fasta $1 --nr_clus_mapped vfs_inAB_res.tsv 
