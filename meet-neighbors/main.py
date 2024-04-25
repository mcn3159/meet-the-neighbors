import subprocess
import argparse
import os
import pandas as pd
import warnings
import dask
import dask.bag as db
import dask.dataframe as dd
import time
import pickle as pkl
import glob


import neighbors_frm_mmseqs as n
import plot_map_neighborhood_res as pn
import compare_neighborhoods as c
import glm_input_frm_neighbors as glm
import compute_umap as cu

def get_parser():
    parser = argparse.ArgumentParser("neighbors",argument_default=argparse.SUPPRESS,description="Meet-the-neighbors extracts and analyzes genomic neighborhoods and runs analyses from protein fastas and their respective gffs",epilog="Madu Nzerem 2023")
    subparsers = parser.add_subparsers(help='Sub-command options',dest="subcommand")
    extract_neighbors = subparsers.add_parser("extract_neighbors",help="Extract neighborhoods from fastas/gffs")
    extract_neighbors.add_argument("--query_fasta","-qf", type=str, required=True, default=None, help="Proteins to search and find neighborhoods for")
    extract_neighbors.add_argument("--seq_id","-s", type=float, required=False, default=0.9, help="Sequence identity for mmseqs search in genomes")
    extract_neighbors.add_argument("--cov","-c", type=float, required=False, default=0.9, help="Sequence coverage for mmseqs search in genomes")
    extract_neighbors.add_argument("--mem","-m", type=int, required=False, default=8, help="Memory")
    extract_neighbors.add_argument("--threads", type=int,default=4, help="Number of threads")

    extract_neighbors.add_argument("--genomes","-g", type=str, required=True, help="Give path to folder w/ proteins and gffs")
    extract_neighbors.add_argument("--out","-o", type=str, required=True, default=None, help="Output directory")
    extract_neighbors.add_argument("--genomes_db","-g_db", type=str, required=False, default=None, help="Path to dir containing mmseqs genomesdb with name of db. Like dir/to/path/genomesDB")
    extract_neighbors.add_argument("--test_fastas", type=str, required=False, default=None, help="Run with test fastas?")
    extract_neighbors.add_argument("--fasta_per_neighborhood", required=False, type=str, default=None, help="To get one fasta per neighborhood")
    extract_neighbors.add_argument("--from_vfdb","-v",required=False,action="store_true",default=None,help="Indicate if search queries are solely from vfdb, to then group by their vf_name")
    extract_neighbors.add_argument("--min_hits","-mih",required=False,type=int,default=5,help="Minimum number of genomes required to report neighborhood")
    extract_neighbors.add_argument("--resume","-r",required=False,action="store_true",help="Resume where program Neighbors left off. Output directory must be the same")
    extract_neighbors.add_argument("--glm",required=False,action="store_true",help="Create output formatted for glm input.")
    extract_neighbors.add_argument("--glm_threshold",type=str,default=0.95,required=False,help="Sets threshold for the minimal percent difference between neighborhoods to be returned, for a given query")
    extract_neighbors.add_argument("--plot","-p",action="store_true", required=False, default=None, help="Plot data")
    extract_neighbors.add_argument("--plt_from_saved","-pfs",type=str, required=False, default=None, help="Plot from a saved neighborhood tsv")
    extract_neighbors.add_argument("--neighborhood_size","-ns",type=int, required=False, default=20000, help="Size in bp of neighborhood to extract")
    extract_neighbors.add_argument("--min_prots","-mip",type=int, required=False, default=3, help="Minimum number of proteins in neighborhood")
    extract_neighbors.add_argument("--max_prots","-map",type=int, required=False, default=30, help="Maximum number of proteins in neighborhood")
    extract_neighbors.add_argument("--red_olp",required=False,action="store_true",help="Reduce amount of overlapping neighborhoods. Default 10kb. Not suggested for glm inputs")
    extract_neighbors.add_argument("-ho","--head_on",required=False,action="store_true",help="Extract neighborhoods with genes in opposite orientations")

    comp_neighbors = subparsers.add_parser("compare_neighborhoods",help="Compare multiple neighborhood tsvs")
    comp_neighbors.add_argument('--neighborhood1','-n1',type=str,required=True,help="Give full path to 1st neighborhood to compare")
    comp_neighbors.add_argument('--neighborhood2','-n2',type=str,required=True,help="Give full path to 2nd neighborhood to compare")
    comp_neighbors.add_argument('--name1',type=str,required=True,help="Name to give neighborhood1")
    comp_neighbors.add_argument('--name2',type=str,required=True,help="Name to give neighborhood2")
    comp_neighbors.add_argument('--out','-o',type=str,default='',required=False,help="Output directory")

    compute_umap = subparsers.add_parser("compute_umap",help="Compute umap from glm_outputs")
    compute_umap.add_argument("--glm_out",type=str,required=True,help="Give directory containing glm embeddings")
    compute_umap.add_argument("--neighborhood_run",type=str,required=True,help="Give directory containing neighborhoods used for glm inputs")
    compute_umap.add_argument("--umap_obj",type=str,required=False,help="Path to umap object file")
    compute_umap.add_argument("--embedding_df",type=str,required=False,help="Path to embedding tsv dataframe")
    compute_umap.add_argument("--umap_name",type=str,required=False,default="umap",help="Filename for umap plot")
    compute_umap.add_argument('--out','-o',type=str,default='',required=False,help="Output directory")
    compute_umap.add_argument('--label','-l',type=str,default='vf_category',required=False,help="Column label to color umap points by. Current options are vf_category,vf_name,vf_subcategory,vfdb_species,vfdb_genus,vf_id")
    compute_umap.add_argument('--width',type=int,default=1000,required=False,help="Width of umap plot")
    compute_umap.add_argument('--legend',action="store_true",required=False,help="Show legend on umap plot")
    return parser

def check_dirs(*args):
    return [dir if dir[-1]=="/" else dir+"/" for dir in args]

def run(parser):
    start_time = time.time()
    args = parser.parse_args()
    if args.subcommand == "extract_neighbors":
        assert os.path.isdir(args.out), "Output directory not found"
        dirs_l = check_dirs(args.genomes,args.out)
        args.genomes,args.out = dirs_l[0],dirs_l[1]
        faa_dir = f"{args.genomes}*.faa"
        if args.genomes_db: #not sure how to set this argparse paremeter default to whatever args.genomes is
            genomes_db = args.genomes_db
        else:
            genomes_db = f"{args.out}genomesDB"

        if not args.plt_from_saved:
            if not args.genomes_db:
                if (not os.path.isfile(genomes_db) and args.resume) or (not args.resume):
                    subprocess.run(f"mmseqs createdb {faa_dir} {genomes_db}",shell=True,check=True)

            if (not os.path.isfile(f"{args.out}queryDB") and args.resume) or (not args.resume):
                subprocess.run(["mmseqs","createdb",args.query_fasta,f"{args.out}queryDB"],check=True)

            if (not os.path.isfile(f"{args.out}vfs_in_genomes.tsv") and args.resume) or (not args.resume):
                subprocess.run(f"mmseqs search {args.out}queryDB {genomes_db} {args.out}vfs_in_genomes {args.out}tmp_search --min-seq-id {args.seq_id} --cov-mode 0 -c {args.cov} --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} --start-sens 1 --sens-steps 3 -s 7"
        ,shell=True,check=True)
                subprocess.run(["mmseqs", "convertalis", f"{args.out}queryDB", f"{genomes_db}", f"{args.out}vfs_in_genomes", f"{args.out}vfs_in_genomes.tsv", "--format-output", "query,target,evalue,pident,qcov,fident,alnlen,qheader,theader,tset,tsetid"] 
        ,check=True)
            
            if (not os.path.isfile(f"{args.out}combined_fastas_clust_res.tsv") and args.resume) or (not args.resume):
                mmseqs_grp_db,mmseqs_search = n.read_search_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}vfs_in_genomes.tsv",threads=args.threads)
                neighborhood_db = db.map(n.get_neigborhood,mmseqs_grp_db,args)
                neighborhood_db = neighborhood_db.flatten()
                n.run_fasta_from_neighborhood(dir_for_fasta=args.genomes,neighborhood=neighborhood_db,
                                            fasta_per_uniq_neighborhood=args.fasta_per_neighborhood,out_folder=args.out,test=args.test_fastas,threads=args.threads)
                subprocess.run(f"mmseqs createdb {args.out}combined_fasta_partition* {args.out}combined_fastas_db",shell=True,check=True)
                subprocess.run(f"mmseqs linclust {args.out}combined_fastas_db {args.out}combined_fastas_clust --cov-mode 0 -c {args.cov} --min-seq-id {args.seq_id} --similarity-type 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} {args.out}tmp_clust",
                                shell=True,check=True)
                subprocess.run(f"mmseqs createtsv {args.out}combined_fastas_db {args.out}combined_fastas_db {args.out}combined_fastas_clust {args.out}combined_fastas_clust_res.tsv"
        ,shell=True,check=True)
                
            if (len(glob.glob(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv"))==0 and args.resume) or (not args.resume):
                mmseqs_grp_db,mmseqs_search = n.read_search_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}vfs_in_genomes.tsv",threads=args.threads)
                mmseqs_search = dd.from_pandas(mmseqs_search,npartitions=args.threads) # make it a dask dataframe there instad of in read_search_tsv() b/c its much easier to run
                mmseqs_clust = pn.prep_cluster_tsv(f"{args.out}combined_fastas_clust_res.tsv")
                if args.red_olp:
                    mmseqs_groups = list(mmseqs_clust.groupby(['gff', 'strand', 'seq_id'])) #cant groupby on its own with dask
                    mmseqs_groups = db.from_sequence(mmseqs_groups,npartitions=args.threads)
                    mmseqs_clust = db.map(pn.reduce_overlap,mmseqs_groups,window=10000)
                    mmseqs_clust = pd.concat(mmseqs_clust.compute())
                print(f"!!! Clustering df size after removing overlapping neighborhoods: {mmseqs_clust.shape} !!!")
                mmseqs_clust = pn.map_vfcenters_to_vfdb_annot(mmseqs_clust,mmseqs_search,args.from_vfdb)
                
                subprocess.run(f"mkdir {args.out}clust_res_in_neighborhoods",shell=True,check=True)
                mmseqs_clust.to_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv",index=False,sep="\t")
                mmseqs_clust = mmseqs_clust.compute()
            
            elif len(glob.glob(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv"))>0 and args.resume:
                mmseqs_clust = dd.read_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv",sep="\t")
                mmseqs_clust = mmseqs_clust.compute() #reading with dask then computing is usually faster than read w/ pandas
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            cluster_neighborhoods_by = "query"
            class_objs = {vf:pn.VF_neighborhoods(cdhit_sub_vf=mmseqs_clust[mmseqs_clust[cluster_neighborhoods_by]==vf],dbscan_eps=0.15,dbscan_min=3)
                        for vf in set(mmseqs_clust[cluster_neighborhoods_by])}
            neighborhood_plt_df = pd.DataFrame.from_dict([class_objs[n].to_dict() for n in class_objs])
            neighborhood_plt_df = neighborhood_plt_df[neighborhood_plt_df['total_hits']>args.min_hits]
            neighborhood_plt_df['bubble_size'] = (100/(neighborhood_plt_df['noise']+1)).astype(int)
            
            if args.from_vfdb:
                # add vf info to neighborhood queries for glm embed color coding and other plotting
                mmseqs_clust_formerge = mmseqs_clust.drop_duplicates(subset=["query"])
                neighborhood_plt_df = pd.merge(neighborhood_plt_df,mmseqs_clust_formerge[['query','vf_name','vf_id','vf_subcategory','vf_category','vfdb_species','vfdb_genus']],on='query',how='left')

            neighborhood_plt_df.to_csv(f"{args.out}neighborhood_results_df.tsv",sep='\t',index=False,header=True)

            if args.glm:
                glm_input_out = f"glm_inputs/"
                subprocess.run(f"mkdir {args.out}{glm_input_out}",shell=True,check=True) #should return an error if the path already exists, don't want to make duplicates
                subprocess.run(f"mmseqs createsubdb {args.out}combined_fastas_clust {args.out}combined_fastas_db {args.out}combined_fastas_clust_rep",shell=True,check=True)
                subprocess.run(f"mmseqs convert2fasta {args.out}combined_fastas_clust_rep {args.out}combined_fastas_clust_rep.fasta",shell=True,check=True)
                print("!!!Grabbing glm inputs!!!")

                uniq_neighborhoods_d = {query:class_objs[query].get_neighborhood_names(args.glm_threshold) for query in class_objs}
    
                db.map(glm.get_glm_input,query=db.from_sequence(uniq_neighborhoods_d.keys(),npartitions=args.threads),
                       uniq_neighborhoods_d=uniq_neighborhoods_d,neighborhood_res=neighborhood_plt_df,mmseqs_clust=mmseqs_clust,args=args).persist()
        elif args.plt_from_saved:
            neighborhood_plt_df = pd.read_csv(args.plt_from_saved,sep='\t')
        if args.plot:
            pn.plt_neighborhoods(neighborhood_plt_df,args.out,vfdb=args.from_vfdb)
            pn.plt_hist_neighborh_clusts(neighborhood_plt_df,args.out)
            pn.plt_regline_scatter(neighborhood_plt_df,args.out)
            pn.plt_box_entropy(neighborhood_plt_df,out=args.out,vfdb=args.from_vfdb)
    
    if args.subcommand == "compare_neighborhoods":
        dirs_l = check_dirs(args.out)
        args.out = dirs_l[0]
        neighborhood1,neighborhood2 = pd.read_csv(args.neighborhood1,sep='\t'),pd.read_csv(args.neighborhood2,sep='\t')
        c.compare_neighborhood_entropy(neighborhood1,neighborhood2,label1=args.name1,label2=args.name2,out=args.out)
        c.compare_uniqhits_trends(neighborhood1,neighborhood2,label1=args.name1,label2=args.name2,out=args.out,write_table=True)

    if args.subcommand == "compute_umap":
        if len(args.out) > 1:
            dirs_l = check_dirs(args.neighborhood_run,args.glm_out,args.out)
            neighborhood_dir,glm_out,args.out = dirs_l[0],dirs_l[1],dirs_l[2]
        else:
            dirs_l = check_dirs(args.neighborhood_run,args.glm_out)
            neighborhood_dir,glm_out = dirs_l[0],dirs_l[1]

        mmseqs_clust = dd.read_csv(f"{neighborhood_dir}clust_res_in_neighborhoods/mmseqs_clust_*.tsv",sep="\t")
        mmseqs_clust = mmseqs_clust.compute()

        umapper,embedding_df_merge = args.umap_obj,args.embedding_df
        if (not args.umap_obj) and (not args.embedding_df):
            embedding_df = cu.unpack_embeddings(glm_out,mmseqs_clust)
            umapper,embedding_df_merge = cu.get_glm_umap_df(embedding_df,mmseqs_clust)
            handle = open("umapper.obj","wb")
            pkl.dump(umapper,handle)
            handle.close()
        else:
            handle = open(args.umap_obj,"rb")
            umapper = pkl.load(handle)
            handle.close()
            embedding_df_merge = pd.read_csv(args.embedding_df,sep="\t")
            
        cu.plt_baby(umapper,embedding_df_merge,plt_name=args.umap_name,outdir=args.out,
                    legend=args.legend,width=args.width,label=args.label)

    print(f"Done! Took --- %s seconds --- to complete" % (time.time() - start_time))
    return

if __name__ == "__main__":
    parser = get_parser()
    run(parser)
    quit()

