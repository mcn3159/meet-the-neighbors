import pandas as pd
import sys
from Bio import SeqIO
import argparse
import re
import dask
import dask.bag as db
from dask.dataframe import from_pandas
import _pickle as cPickle
import time

from process_gffs import gff2pandas
from process_gffs import get_protseq_frmFasta

def read_mmseqs_tsv(**kwargs):
    #read in mmseqs_tsv
    #create vf info col
    #send to dask df
    #return df grouped by target filename
    ##PARAMS
    vfdb = kwargs.get("vfdb",None)
    mmseqs_search_res = kwargs.get("input_mmseqs",None)
    headers = ['query','target','evalue','pident','qcov','fident','alnlen','qheader','theader','tset','tsetid']
    if vfdb:
        #extract vf annots from columns
        mmseqs = pd.read_csv("vfs_inOrgs.tsv",sep='\t',names=headers)
        pattern = r"(\) .* \[)([^\]]+)\s*\(([^)]+)\)\s*-\s*([^\]]+)\(" #use https://regex101.com/ to see what pattern is doing
        mmseqs[['vf_name','vf_subcategory','vf_id','vf_category']] = mmseqs['qheader'].str.extract(pattern)
        pattern = r"\) (.+?) \["
        mmseqs['vf_name'] = mmseqs.vf_name.str.extract(pattern)
        mmseqs.to_csv("vfs_inOrgs.tsv",sep='\t',index=False)
    elif mmseqs_search_res:
        mmseqs = pd.read_csv(mmseqs_search_res,sep='\t',names=headers)
        mmseqs.to_csv(mmseqs_search_res,index=False,headers=True,sep='\t')
    mmseqs_grp = mmseqs.groupby('tset') # grouped vf hits by fasta where hits were found
    mmseqs_l = list(mmseqs_grp) #turning it into a list so I can send it to a dask bag
    #turning into list b/c dask_df_apply(func) is a literal pain in the ass to use
    mmseqs_grp_db = db.from_sequence(mmseqs_l,npartitions=partitions)
    print(f'!!!{partitions} threads detected!!!')
    #mmseqs_dask = mmseqs_dask.groupby('tset').persist()
    return mmseqs_grp_db

# mmseq hits to a dictionary where keys are vf_centers? values are seq_id,strand,and gff where it was found
# dictionary to dask bag, along w/ mmseqs tsv, then groupby mmseqs tsv by

def get_neigborhood(mmseqs_group,dir_for_gffs,window):
    # condition: What if the same protein appears twice in a genome, but has different neighborhoods? They should have the same query
    gff = dir_for_gffs+'/'+mmseqs_group[1].tset.iloc[0].split('protein.faa')[0]+'genomic.gff'
    gff_df = gff2pandas(gff)
    #gff_df = from_pandas(gff_df,npartitions=partitions)
    vf_centers = gff_df[gff_df['protein_id'].isin(list(mmseqs_group[1].target))] # maybe i dont need the list command
    #print(f"{len(vf_centers)} hits found in {gff.split('/')[-1]}")
    neighborhoods = []
    for row in vf_centers.itertuples():
        gff_df_strand = gff_df[(gff_df['strand'] == row.strand) & (gff_df['seq_id'] == row.seq_id)]
        neighborhood_df = gff_df_strand.copy()
        neighborhood_df = neighborhood_df[
            (neighborhood_df['start'] >= row.start - window) &
            (neighborhood_df['end'] <= row.end + window)]
        neighborhood_df['VF_center'] = row.protein_id
        neighborhood_df['gff_name'] = gff.split('/')[-1].split('_genomic.gff')[0]

        if len(neighborhood_df) < 2: #filter out really small neighborhoods
            print(f"VF Neighborhood {row.protein_id} from gff {gff.split('/')[-1]} filtered out because there are less than 2 proteins")
            continue
        neighborhoods.append(neighborhood_df)
    return neighborhoods

def run_fasta_from_neighborhood(dir_for_fasta,neighborhood,**kwargs):
    # Create protein fasta from neighborhoods
    #modify this func to output fastas in a separate directory
    start_time = time.time()
    test = kwargs.get("test",None)
    out_folder = kwargs.get("out_folder","") # the alternative is blank so that the outdir is the current working directory
    fasta_per_neighborhood = kwargs.get("fasta_per_neighborhood",None)
    if test:
        file = open(test,"rb")
        neighborhood = cPickle.load(file)
        file.close()
        print("!!!Neighborhoods loaded from pickle!!!")
        neighborhood = db.from_sequence(neighborhood,npartitions=partitions)
        #uncomment below if pikled object has empty neighborhoods
        #neighborhood = neighborhood.filter(lambda x: x is not None) 
        #neighborhood = neighborhood.filter(lambda x: len(x)>0)
        #neighborhood = neighborhood.flatten().persist()

    run_fasta = db.map(get_protseq_frmFasta,dir_for_fasta,neighborhood,to_fasta=fasta_per_neighborhood)
    if fasta_per_neighborhood:
        run_fasta.compute()
        print(f"--- %s seconds --- for creating fastas" % (time.time() - start_time))
        return
    run_fasta = run_fasta.flatten().repartition(partitions)
    if len(out_folder)>1: out_folder + '/'
    run_fasta.to_textfiles(f"{out_folder}combined_fasta_partition*.faa")
    print(f"--- %s seconds --- for creating fastas" % (time.time() - start_time))
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="full dir of .sh file calling mmseqs")
    parser.add_argument("--genomes", type=str, required=True, default=None, help="Give path to folder w/ gffs")
    parser.add_argument("--threads", type=int,default=None, help="Number of threads")
    parser.add_argument("--out", type=str, required=True, default=None, help="Output directory")
    parser.add_argument("--test_fastas", type=str, required=False, default=None, help="Run with test fastas?")
    parser.add_argument("--fasta_per_neighborhood", required=False, type=str, default=None, help="To get one fasta per neighborhood")
    parser.add_argument("--frm_vfdb",required=False,action="store_true",help="Indicate if search queries are solely from vfdb")
    parser.add_argument("--mmseqs_search_res",required=False,type=str,help="If not using a vfdb search, input custom mmseqs search")
    args = parser.parse_args()

    partitions = args.threads
    #col_headers,mmseqs_tsv = extract_string_from_sh(args.mmseqs_sh) hard coded these params, since they shouldnt be changing
    mmseqs_grp_db = read_mmseqs_tsv(vfdb=args.frm_vfdb,input_mmseqs=args.mmseqs_search_res)
    neighborhood_db = db.map(get_neigborhood,mmseqs_grp_db,args.genomes,10000)
    neighborhood_db = neighborhood_db.flatten()
    run_fasta_from_neighborhood(dir_for_fasta=args.genomes,neighborhood=neighborhood_db,
                                fasta_per_neighborhood=args.fasta_per_neighborhood,out_folder=args.out,test=args.test_fastas)
        
#call gffpandas, load in gff df for group, subset for vf ids, loop thru that subset to return neighborhoods, to return neighborhood...
#group df by strand and contig as call in iteration for vf_id subset, then use code to get matching_rows variable found in process_gffs
#loop thru each unique tag that is in particular gff
#return individual neighborhood dfs that are subsets of gffs



