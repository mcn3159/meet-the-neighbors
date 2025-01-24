# import pandas as pd
import argparse
import re
import dask
import dask.bag as db
import os
import glob
import tempfile

# from process_gffs import gff2pandas
# from process_gffs import get_protseq_frmFasta
import meetneighbors.process_gffs as pg

def read_search_tsv(**kwargs):
    #read in mmseqs_tsv
    #create vf info col
    #send to dask df
    #return df grouped by target filename
    ##PARAMS
    vfdb = kwargs.get("vfdb",None)
    mmseqs_search_res = kwargs.get("input_mmseqs",None)
    headers = ['query','target','evalue','pident','qcov','fident','alnlen','qheader','theader','tset','tsetid']
    partitions = kwargs.get("threads",4)
    mmseqs = dask.dataframe.read_csv(mmseqs_search_res,sep='\t')
    mmseqs = mmseqs.compute()
    if (mmseqs.columns[-1] != 'tsetid') and (mmseqs.columns[-1] != 'vfdb_genus'): mmseqs.columns = headers
    if vfdb:
        #extract vf annots from columns
        pattern = r"(\) .* \[)([^\]]+)\s*\(([^)]+)\)\s*-\s*([^\]]+)\(" #use https://regex101.com/ to see what pattern is doing
        mmseqs[['vf_name','vf_subcategory','vf_id','vf_category']] = mmseqs['qheader'].str.extract(pattern)
        pattern = r"\) (.+?) \["
        mmseqs['vf_name'] = mmseqs.vf_name.str.extract(pattern)
        pattern = r"\[[^\]]+\]\s*\[([^\]]+)\]"
        mmseqs['vfdb_species'] = mmseqs['qheader'].str.extract(pattern)
        mmseqs['vfdb_genus'] = mmseqs.vfdb_species.str.split(' ').str[0]

    #mmseqs.to_csv(mmseqs_search_res,index=False,header=True,sep='\t')
    mmseqs_grp = mmseqs.groupby('tset') # grouped vf hits by fasta where hits were found
    mmseqs_l = list(mmseqs_grp) #turning it into a list so I can send it to a dask bag
    #turning into list b/c dask_df_apply(func) is a literal pain in the ass to use
    mmseqs_grp_db = db.from_sequence(mmseqs_l,npartitions=partitions)
    #mmseqs_dask = mmseqs_dask.groupby('tset').persist()
    return mmseqs_grp_db,mmseqs


def get_neigborhood(logger,args,tmpd,**kwargs):
    # this function gets neighborhoods of proteins based on position denoted in gff_df, filters out too big and too small neighborhoods
    # condition: If the same protein appears twice in a genome, but has different neighborhoods? They should have the same query

    mmseqs_group = kwargs.get("mmseqs_groups",None)
    genome_query = kwargs.get("genome_query",None)
    args.head_on = kwargs.get("head_on",None) # to be compatible with chop genome

    if mmseqs_group:  
        gff = args.genomes + mmseqs_group[1].tset.iloc[0].split('protein.faa')[0] + 'genomic.gff'
        gff_df = pg.gff2pandas(gff)
        protein_ids = list(mmseqs_group[1].target)
    
    if genome_query:
        gff = args.genomes + genome_query + '.gff'
        gff_df = pg.gff2pandas(gff)
        gff_df['protein_id'] = gff_df['protein_id'].str.split(':').str[-1] # remove weird characters in protein id

        # check that we have a protein file with an appropriate suffix
        protein_file = gff.split('.gff')[0] + ".faa"
        if not os.path.isfile(protein_file):
            protein_file = gff.split('.gff')[0] + "fasta"
            assert os.path.isfile(protein_file), f"Protein file for {genome_query} with .faa or .fasta suffix not found"

        # get list of protein ids to then use on gff,subset protein id to match what's in gff_df
        protein_ids = set([rec.id.split('|')[-1] for rec in pg.SeqIO.parse(protein_file,"fasta")]) 

    # subset gff for protein centers that we're interested in
    # logger.warning(f"Before size of gffdf: {gff_df.shape}")
    vf_centers = gff_df[gff_df['protein_id'].isin(protein_ids)]
    # logger.warning(f"After size of gffdf: {vf_centers.shape}")

    window = args.neighborhood_size/2
    neighborhoods = []
    removed_neighborhoods = []
    max_neighbors_report = 2
    report = 0
    for row in vf_centers.itertuples(): # this adds a new neighborhood (neighborhood_df) to a list (neighborhoods)

        # subset the genome (gff_df) for genes on the same contig and/or strand
        if args.head_on:
            gff_df_strand = gff_df[gff_df['seq_id'] == row.seq_id]
        else:
            gff_df_strand = gff_df[(gff_df['strand'] == row.strand) & (gff_df['seq_id'] == row.seq_id)]
        neighborhood_df = gff_df_strand.copy()

        # further subset the genome (now neighborhood_df) for genes within the predefined neighborhood limits default +/- 20kb
        neighborhood_df = neighborhood_df[
            (neighborhood_df['start'] >= row.start - window) &
            (neighborhood_df['end'] <= row.end + window)]
        neighborhood_df['VF_center'] = row.protein_id
        neighborhood_df['gff_name'] = gff.split('/')[-1].split('_genomic.gff')[0].split('.gff')[0] # for compatibility with chop_genomes

        # remove neighborhoods that don't fit specified conditions
        if len(neighborhood_df) < args.min_prots: #neighborhood centers could be near a contig break causing really small neighborhoods, which isnt helpful info, remove these
            removed_neighborhoods.append(neighborhood_df['VF_center'].iloc[0] + '!!!' + neighborhood_df['gff_name'].iloc[0]) # save this info for debugging later
            if report < max_neighbors_report:
                logger.warning(f"Neighborhood {row.protein_id} from gff {gff.split('/')[-1]} filtered out because there are less than {args.min_prots} proteins") #maybe I should output this type of info to a text file
                report +=1
            continue
        if len(neighborhood_df) > args.max_prots: # glm can handle up to 30 proteins
            removed_neighborhoods.append(neighborhood_df['VF_center'].iloc[0] + '!!!' + neighborhood_df['gff_name'].iloc[0])
            if report < max_neighbors_report:
                logger.warning(f"Neighborhood {row.protein_id} from gff {gff.split('/')[-1]} filtered out because there are more than {args.max_prots} proteins")
                report +=1
            continue
        neighborhoods.append(neighborhood_df)
    
    # save removed neighborhoods to a temporary text file for each partition, for analysis later
    
    tmpf = tempfile.NamedTemporaryFile(mode='w+t',prefix='protgff_',suffix='.txt',dir=tmpd,delete=False)
    tmpf.writelines(f"{prot_gff}\n" for prot_gff in removed_neighborhoods)
    tmpf.close()
    return neighborhoods

def run_fasta_from_neighborhood(logger,args,dir_for_fasta,neighborhood,**kwargs):
    # Create protein fasta from neighborhoods
    #modify this func to output fastas in a separate directory
    partitions = kwargs.get("threads",4)
    out_folder = kwargs.get("out_folder","") # the alternative is blank so that the outdir is the current working directory
    fasta_per_neighborhood = kwargs.get("fasta_per_neighborhood",None)

    seq_recs = db.map(pg.get_protseq_frmFasta,logger,args,dir_for_fasta,neighborhood,fasta_per_neighborhood=fasta_per_neighborhood)
    seq_recs.flatten().repartition(partitions).to_textfiles(f"{out_folder}combined_fasta_partition*.faa")
    return
        
#call gffpandas, load in gff df for group, subset for vf ids, loop thru that subset to return neighborhoods, to return neighborhood...
#group df by strand and contig as call in iteration for vf_id subset, then use code to get matching_rows variable found in process_gffs
#loop thru each unique tag that is in particular gff
#return individual neighborhood dfs that are subsets of gffs



