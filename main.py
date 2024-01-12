import subprocess
import argparse
import os
import pandas as pd
import warnings
import dask
import dask.bag as db

import neighbors_frm_mmseqs as n
import plot_map_neighborhood_res as pn
import compare_neighborhoods as c

def get_parser():
    #next add sub command for entropies
    parser = argparse.ArgumentParser("neighbors",description="Meet-the-neighbors extracts genomic neighborhoods and runs analyses on them from protein fastas and their respective gffs",epilog="Madu Nzerem 2023")
    parser.add_argument("--query_fasta","-qf", type=str, required=True, default=None, help="Proteins to search and find neighborhoods for")
    parser.add_argument("--seq_id","-s", type=float, required=False, default=0.9, help="Sequence identity for mmseqs search in genomes")
    parser.add_argument("--cov","-c", type=float, required=False, default=0.9, help="Sequence coverage for mmseqs search in genomes")
    parser.add_argument("--mem","-m", type=int, required=False, default=8, help="Memory")
    parser.add_argument("--threads", type=int,default=4, help="Number of threads")

    parser.add_argument("--genomes","-g", type=str, required=True, default=None, help="Give path to folder w/ proteins and gffs")
    parser.add_argument("--out","-o", type=str, required=True, default=None, help="Output directory")
    parser.add_argument("--test_fastas", type=str, required=False, default=None, help="Run with test fastas?")
    parser.add_argument("--fasta_per_neighborhood", required=False, type=str, default=None, help="To get one fasta per neighborhood")
    parser.add_argument("--from_vfdb","-v",required=False,action="store_true",help="Indicate if search queries are solely from vfdb, to then group by their vf_id")
    #parser.add_argument("--mmseqs_search_res",required=False,type=str,help="If not using a vfdb search, input custom mmseqs search")
    parser.add_argument("--min_hits","-min",required=False,type=int,default=5,help="Minimum number of hits required to report neighborhood")
    parser.add_argument("--resume","-r",required=False,action="store_true",help="Resume where program Neighbors left off. Output directory must be the same")

    #parser.add_argument("--mmseqs_tsv", type=str, required=True, default=None, help="Give mmseqs clustered tsv")
    parser.add_argument("--plot","-p",action="store_true", required=False, default=None, help="Plot data")
    parser.add_argument("--plt_from_saved","-pfs",type=str, required=False, default=None, help="Plot from a saved neighborhood tsv")

    subparsers = parser.add_subparsers(help='Sub-command options',dest="subcommand")
    comp_neighbors = subparsers.add_parser("compare_neighborhoods",help="Functionality to compare multiple neighborhood tsvs")
    comp_neighbors.add_argument('--neighborhood1','-n1',type=str,required=True,help="Give full path to 1st neighborhood to compare")
    comp_neighbors.add_argument('--neighborhood2','-n2',type=str,required=True,help="Give full path to 2nd neighborhood to compare")
    comp_neighbors.add_argument('--name1',type=str,required=True,help="Name to give neighborhood1")
    comp_neighbors.add_argument('--name2',type=str,required=True,help="Name to give neighborhood2")

    return parser

def check_dirs(*args):
    return [dir if dir[-1]=="/" else dir+"/" for dir in args]

def run(parser):
    args = parser.parse_args()
    

    if __name__ == "__main__":
        assert os.path.isdir(args.out), "Output directory not found"
        #subprocess actually runs really slow, just realized its b/c of sshfs
        dirs_l = check_dirs(args.genomes,args.out)
        args.genomes,args.out = dirs_l[0],dirs_l[1]
        faa_dir = f"{args.genomes}*.faa"

        if not args.plt_from_saved:

            if not os.path.isfile(f"{args.out}genomesDB") and args.resume:
                subprocess.run(["mmseqs","createdb",args.query_fasta,f"{args.out}queryDB"],check=True)
                subprocess.run(f"mmseqs createdb {faa_dir} {args.out}genomesDB",shell=True,check=True)

            if not os.path.isfile(f"{args.out}vfs_in_genomes.tsv") and args.resume:
                subprocess.run(f"mmseqs search {args.out}queryDB {args.out}genomesDB {args.out}vfs_in_genomes {args.out}tmp_search --min-seq-id {args.seq_id} --cov-mode 0 -c {args.cov} --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} --start-sens 1 --sens-steps 3 -s 7"
        ,shell=True,check=True)
                subprocess.run(["mmseqs", "convertalis", f"{args.out}queryDB", f"{args.out}genomesDB", f"{args.out}vfs_in_genomes", f"{args.out}vfs_in_genomes.tsv", "--format-output", "query,target,evalue,pident,qcov,fident,alnlen,qheader,theader,tset,tsetid"] 
        ,check=True)
                
            if not os.path.isfile(f"{args.out}combined_fastas_clust_res.tsv") and args.resume:
                mmseqs_grp_db,mmseqs_search = n.read_mmseqs_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}vfs_in_genomes.tsv")
                neighborhood_db = db.map(n.get_neigborhood,mmseqs_grp_db,args.genomes,10000)
                neighborhood_db = neighborhood_db.flatten()
                n.run_fasta_from_neighborhood(dir_for_fasta=args.genomes,neighborhood=neighborhood_db,
                                            fasta_per_neighborhood=args.fasta_per_neighborhood,out_folder=args.out,test=args.test_fastas)
                subprocess.run(f"mmseqs createdb {args.out}combined_fasta_partition* {args.out}combined_fastas_db",shell=True,check=True)
                subprocess.run(f"mmseqs linclust {args.out}combined_fastas_db {args.out}combined_fastas_clust --cov-mode 0 -c {args.cov} --min-seq-id {args.seq_id} --similarity-type 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} {args.out}tmp_clust",
                                shell=True,check=True)
                subprocess.run(f"mmseqs createtsv {args.out}combined_fastas_db {args.out}combined_fastas_db {args.out}combined_fastas_clust {args.out}combined_fastas_clust_res.tsv"
        ,shell=True,check=True)
                
            elif os.path.isfile(f"{args.out}combined_fastas_clust_res.tsv") and args.resume:
                mmseqs_grp_db,mmseqs_search = n.read_mmseqs_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}vfs_in_genomes.tsv")
            mmseqs_clust = pn.prep_mmseqs_tsv(f"{args.out}combined_fastas_clust_res.tsv")
            mmseqs_clust = pn.map_vfcenters_to_vfdb_annot(mmseqs_clust,mmseqs_search,vfdb=args.from_vfdb)
            cluster_neighborhoods_by = "query"
            if args.from_vfdb:
                cluster_neighborhoods_by = "vf_id"

            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            class_objs = {vf:pn.VF_neighborhoods(cdhit_sub_vf=mmseqs_clust[mmseqs_clust[cluster_neighborhoods_by]==vf],dbscan_eps=0.15,dbscan_min=3)
                        for vf in set(mmseqs_clust[cluster_neighborhoods_by])}
            neighborhood_plt_df = pd.DataFrame.from_dict([class_objs[n].to_dict() for n in class_objs])

            neighborhood_plt_df = neighborhood_plt_df[neighborhood_plt_df['total_hits']>args.min_hits]
            neighborhood_plt_df['bubble_size'] = (100/(neighborhood_plt_df['noise']+1)).astype(int)
            if args.from_vfdb:
                mmseqs_clust.rename(columns={'query':'vf_query'},inplace=True) #renaming here for clarity and mmseqs orginal tsv uses name query
                mmseqs_clust_for_merge = mmseqs_clust.drop_duplicates(subset=['vf_query'])
                neighborhood_plt_df = pd.merge(neighborhood_plt_df,mmseqs_clust_for_merge[['vf_query','vf_name','vf_id','vf_subcategory','vf_category']],on='vf_query',how='left')

            neighborhood_plt_df.to_csv(f"{args.out}neighborhood_results_df.tsv",sep='\t',index=False,header=True)
        elif args.plt_from_saved:
            neighborhood_plt_df = pd.read_csv(args.plt_from_saved,sep='\t')
        if args.plot:
            pn.plt_neighborhoods(neighborhood_plt_df,args.out)
            pn.plt_hist_neighborh_clusts(neighborhood_plt_df,args.out)
            pn.plt_regline_scatter(neighborhood_plt_df,args.out)
        
        if args.subcommand == "compare_neighborhoods":
            neighborhood1,neighborhood2 = pd.read_csv(args.neighborhood1,sep='\t'),pd.read_csv(args.neighborhood2,sep='\t')
            c.compare_neighborhood_entropy(neighborhood1,neighborhood2,label1=args.name1,label2=args.name2)
            c.compare_uniqhits_trends(neighborhood1,neighborhood2,label1=args.name1,label2=args.name2,out=args.out,write_table=True)
        return
    
parser = get_parser()
run(parser)
print("Done!")
