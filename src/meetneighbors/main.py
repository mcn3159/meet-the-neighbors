import subprocess
import argparse
import os
import shutil
import numpy as np
import pandas as pd
import warnings
import dask
dask.config.set(scheduler='processes')
import dask.bag as db
import dask.dataframe as dd
import time
import pickle as pkl
import glob
import logging
import tempfile
import sys


import meetneighbors.neighbors_frm_mmseqs as n
import meetneighbors.plot_map_neighborhood_res as pn
import meetneighbors.compare_neighborhoods as c
import meetneighbors.glm_input_frm_neighbors as glm
import meetneighbors.compute_umap as cu

from meetneighbors.predictvfs import neighborhood_classification as nc
from meetneighbors.predictvfs import structure_classification as sc
from meetneighbors.predictvfs import clf_models as clf
from meetneighbors.predictvfs import loader as l


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
    extract_neighbors.add_argument("--min_hits","-mih",required=False,type=int,default=0,help="Minimum number of genomes required to report neighborhood")
    extract_neighbors.add_argument("--resume","-r",required=False,action="store_true",help="Resume where program Neighbors left off. Output directory must be the same")
    extract_neighbors.add_argument("--glm",required=False,action="store_true",help="Create output formatted for glm input.")
    extract_neighbors.add_argument("--glm_threshold",type=float,default=0.10,required=False,help="Threshold for the minimal percent difference between neighborhoods to be returned, for a given query. Use 0 to disable this type of neighborhood reduction")
    extract_neighbors.add_argument("--glm_cluster",type=str,default="complete",required=False,help="Sklearn agglomerative clustering linkage method to link similar neighborhoods")
    extract_neighbors.add_argument("--plot","-p",action="store_true", required=False, default=None, help="Plot data")
    extract_neighbors.add_argument("--plt_from_saved","-pfs",type=str, required=False, default=None, help="Plot from a saved neighborhood tsv")
    extract_neighbors.add_argument("--neighborhood_size","-ns",type=int, required=False, default=20000, help="Size in bp of neighborhood to extract. 10kb less than start, and 10kb above end of center DNA seq")
    extract_neighbors.add_argument("--min_prots","-mip",type=int, required=False, default=3, help="Minimum number of proteins in neighborhood")
    extract_neighbors.add_argument("--max_prots","-map",type=int, required=False, default=30, help="Maximum number of proteins in neighborhood")
    extract_neighbors.add_argument("--red_olp",required=False,action="store_true",help="Reduce amount of overlapping neighborhoods. Default 10kb.")
    extract_neighbors.add_argument("--olp_window",required=False,type=int,default=10000,help="Change allowable overlap between neighborhoods.")
    extract_neighbors.add_argument("--multi_vf_nn",required=False,action="store_true",help="Filter neighborhoods found for only those with multiple query hits based on a second mmseqs search")
    extract_neighbors.add_argument("--mvn_seq_id","-mvn_s", type=float, required=False, default=0.3, help="Sequence identity for 2nd mmseqs search in genomes")
    extract_neighbors.add_argument("--mvn_cov","-mvn_c", type=float, required=False, default=0.8, help="Sequence coverage for 2nd mmseqs search in genomes")
    extract_neighbors.add_argument("-ho","--head_on",required=False,action="store_true",help="Extract neighborhoods with genes in opposite orientations")
    extract_neighbors.add_argument("--remove_temp",required=False,action="store_true",help="Remove temp directory created along with all of its contents")
    extract_neighbors.add_argument("-ig","--intergenic",required=False,type=int,help="Set a maximum cutoff for the integenic distance of a neighborhood")

    comp_neighbors = subparsers.add_parser("compare_neighborhoods",help="Compare multiple neighborhood tsvs")
    comp_neighbors.add_argument('--neighborhood1','-n1',type=str,required=True,help="Give full path to 1st neighborhood to compare")
    comp_neighbors.add_argument('--neighborhood2','-n2',type=str,required=True,help="Give full path to 2nd neighborhood to compare")
    comp_neighbors.add_argument('--name1',type=str,required=True,help="Name to give neighborhood1")
    comp_neighbors.add_argument('--name2',type=str,required=True,help="Name to give neighborhood2")
    comp_neighbors.add_argument('--out','-o',type=str,default='',required=False,help="Output directory")

    chop_genome = subparsers.add_parser("chop-genome",help="Chop up genome(s) into neighborhoods")
    chop_genome.add_argument("--genomes","-g", type=str, required=True, help="Give path to folder w/ proteins and gffs")
    chop_genome.add_argument("--red_olp",required=False,action="store_true",help="Reduce amount of overlapping neighborhoods. Default 10kb.")
    chop_genome.add_argument("--olp_window",required=False,type=int,default=10000,help="Change allowable overlap between neighborhoods.")
    chop_genome.add_argument("--neighborhood_size","-ns",type=int, required=False, default=20000, help="Size in bp of neighborhood to extract. 10kb less than start, and 10kb above end of center DNA seq")
    chop_genome.add_argument("--threads", type=int,default=4, help="Number of threads")
    chop_genome.add_argument("--min_prots","-mip",type=int, required=False, default=3, help="Minimum number of proteins in neighborhood")
    chop_genome.add_argument("--max_prots","-map",type=int, required=False, default=30, help="Maximum number of proteins in neighborhood")

    # haven't decided whether or not I want add a functionality that clusters all the protein with the genome before sending to glm input, would potentially be a big speed up

    # chop_genome.add_argument("--cluster",action="store_true", required=False, default=30, help="Cluster neighborhoods from genomes")
    # chop_genome.add_argument("--seq_id","-s", type=float, required=False, default=0.9, help="Sequence identity for mmseqs search in genomes")
    # chop_genome.add_argument("--cov","-c", type=float, required=False, default=0.9, help="Sequence coverage for mmseqs search in genomes")
    chop_genome.add_argument('--out','-o',type=str,default='',required=True,help="Output directory")


    compute_umap = subparsers.add_parser("compute_umap",help="Compute umap from glm_outputs")
    compute_umap.add_argument("--glm_in",type=str,required=False,default="glm_inputs",help="Give directory containing inputs used to generate glm embbeds")
    compute_umap.add_argument("--glm_out",type=str,required=True,help="Give directory containing glm embeddings")
    compute_umap.add_argument("--neighborhood_run",type=str,required=True,help="Give directory containing neighborhoods used for glm inputs")
    compute_umap.add_argument("--umap_obj",type=str,required=False,help="If already computed, provide path to umap object file")
    compute_umap.add_argument("--embedding_df",type=str,required=False,help="If already computed, provide path to embedding tsv dataframe")
    compute_umap.add_argument("--umap_name",type=str,required=False,default="umap",help="Filename for umap plot")
    compute_umap.add_argument("--plt_umap","-p",action="store_true",required=False,help="return a plot of umap, must install umap.plot and its dependencies")
    compute_umap.add_argument('--out','-o',type=str,default='',required=False,help="Output directory")
    compute_umap.add_argument('--label','-l',type=str,default='vf_category',required=False,help="Column label to color umap points by. Current options are vf_category,vf_name,vf_subcategory,vfdb_species,vfdb_genus,vf_id")
    compute_umap.add_argument('--width',type=int,default=1000,required=False,help="Width of umap plot")
    compute_umap.add_argument('--legend',action="store_true",required=False,help="Show legend on umap plot")

    predictvf = subparsers.add_parser("predictvf",help="Predict functional categories of proteins given a their gff files and resprective protein fastas")
    predictvf.add_argument("--genomes","-g", type=str, required=True, help="Give path to folder w/ proteins and gffs")
    predictvf.add_argument("--threads", type=int,default=4, help="Number of threads")
    predictvf.add_argument("--ngpus", type=int,default=0, help="Number of gpus to use for embedding making")
    predictvf.add_argument("--glm_embeds",type=str,required=False,help="tsv file containing glm embeddings")
    predictvf.add_argument("--foldseek_structs",type=str,required=True,help="Directory containing foldseek db of query protein structures")
    predictvf.add_argument("--out","-o",type=str,help="Output directory")
    predictvf.add_argument("--qtmcutoff",type=int,default=0.6,help="qTMscore cutoff for defining a structure-based hit, ignores qcovcutoff")
    predictvf.add_argument("--fs_qcovcutoff",type=int,default=0.75,help="foldseek search qcov cutoff for defining a structure-based hit")
    predictvf.add_argument("--lddtcutoff",type=int,default=0.75,help="lddt cutoff for defining a structure-based hit")
    predictvf.add_argument("--include_structhits",type=str,required=False,help="Give path to neighborhood output folder to compute number of structural hits per neighborhood") # should change the name of this arg and make non-optional
    predictvf.add_argument("--neighborhood_size","-ns",type=int, required=False, default=20000, help="Size in bp of neighborhood to extract. 10kb less than start, and 10kb above end of center DNA seq")
    predictvf.add_argument("--min_prots","-mip",type=int, required=False, default=3, help="Minimum number of proteins in neighborhood")
    predictvf.add_argument("--max_prots","-map",type=int, required=False, default=30, help="Maximum number of proteins in neighborhood")
    # predictvf.add_argument("--resume","-r",required=False,action="store_true",help="Resume where program Neighbors left off. Output directory must be the same")

    return parser

def check_dirs(*args):
    return [dir if dir[-1]=="/" else dir+"/" for dir in args]

def get_logger(subcommand,out):
    #logger = logging.basicConfig(filename='log.txt', filemode='w',format='%(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    c_handler,f_handler = logging.StreamHandler(),logging.FileHandler(f"{out}{subcommand}.log")
    f_handler.setLevel(logging.DEBUG)
    c_handler.setLevel(logging.INFO)
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)
    logger.addHandler(c_handler)
    exec_command = f"{' '.join(sys.argv)}"
    logger.debug("Organizing neighborhood meetup...")
    logger.debug(f"the command launched: {exec_command}")
    return logger

def workflow(parser):
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
        logger = get_logger(args.subcommand,args.out)
        logger.debug("Extracting neighborhoods...")
        removed_prot_gffs,tmpd = [], None # used later to check for removed neighborhoods via temp files

        if not args.plt_from_saved:
            if not args.genomes_db:
                if (not os.path.isfile(genomes_db) and args.resume) or (not args.resume):
                    logger.debug("Creating genome database with mmesqs...")
                    subprocess.run(f"mmseqs createdb {faa_dir} {genomes_db} -v 2",shell=True,check=True)
                    
            if (not os.path.isfile(f"{args.out}queryDB") and args.resume) or (not args.resume):
                logger.debug("Creating query database with mmesqs...")
                subprocess.run(["mmseqs","createdb",args.query_fasta,f"{args.out}queryDB","-v","2"],check=True)

            if (not os.path.isfile(f"{args.out}vfs_in_genomes.tsv") and args.resume) or (not args.resume):
                logger.debug("Searching for queries in genome database with mmesqs...")
                subprocess.run(f"mmseqs search {args.out}queryDB {genomes_db} {args.out}vfs_in_genomes {args.out}tmp_search --min-seq-id {args.seq_id} --cov-mode 0 -c {args.cov} -v 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads}  --alignment-mode 3 --start-sens 1 --sens-steps 3 -s 7"
        ,shell=True,check=True)
                subprocess.run(["mmseqs", "convertalis", f"{args.out}queryDB", f"{genomes_db}", f"{args.out}vfs_in_genomes", f"{args.out}vfs_in_genomes.tsv", "--format-output", "query,target,evalue,pident,qcov,fident,alnlen,qheader,theader,tset,tsetid"] 
        ,check=True)
            
            if (not os.path.isfile(f"{args.out}combined_fastas_clust_res.tsv") and args.resume) or (not args.resume):
                mmseqs_grp_db,mmseqs_search = n.read_search_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}vfs_in_genomes.tsv",threads=args.threads)
                logger.info(f"Number of query proteins with hits: {len(set(mmseqs_search['query']))}")

                tmpd = tempfile.mkdtemp(dir=args.out,prefix='tempdir_')

                logger.debug(f"Pulling neighborhoods with {args.threads} threads...")
                neighborhood_db = db.map(n.get_neigborhood,logger,args,tmpd, mmseqs_groups = mmseqs_grp_db,head_on=args.head_on)
                
                tmpfiles = glob.glob(tmpd+'/*') # should probably use pathlib for this or os.path.join?
                for tmpf_path in tmpfiles:
                    with open(tmpf_path,'r') as tmpf:
                        removed_prot_gffs.extend(tmpf.read().splitlines())

                neighborhood_db = neighborhood_db.flatten()
                n.run_fasta_from_neighborhood(logger,args,dir_for_fasta=args.genomes,neighborhood=neighborhood_db,
                                            fasta_per_uniq_neighborhood=args.fasta_per_neighborhood,out_folder=args.out,threads=args.threads)
                logger.debug("Clustering proteins found in all neighborhoods...")
                subprocess.run(f"mmseqs createdb {args.out}combined_fasta_partition* {args.out}combined_fastas_db -v 2",shell=True,check=True)
                # hard coded some clustering params b/c the goal is to reduce redundant proteins
                subprocess.run(f"mmseqs linclust {args.out}combined_fastas_db {args.out}combined_fastas_clust --cov-mode 0 -c 0.90 --min-seq-id 0.90 --similarity-type 2 -v 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads} {args.out}tmp_clust",
                                shell=True,check=True)
                subprocess.run(f"mmseqs createtsv {args.out}combined_fastas_db {args.out}combined_fastas_db {args.out}combined_fastas_clust {args.out}combined_fastas_clust_res.tsv"
        ,shell=True,check=True)
                
                # prepare directory for mmseqs_clust file, here before below if so we don't hit the mkdir error
                subprocess.run(f"mkdir {args.out}clust_res_in_neighborhoods",shell=True,check=True)

            if (len(glob.glob(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv"))==0 and args.resume) or (not args.resume):
                mmseqs_grp_db,mmseqs_search = n.read_search_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}vfs_in_genomes.tsv",threads=args.threads)
                
                # turning mmseqs_search into a dask dataframe was causing an error
                # mmseqs_search = dd.from_pandas(mmseqs_search,npartitions=args.threads) # make it a dask dataframe there instad of in read_search_tsv() b/c its much easier to run
                # logger.debug("Reading in dataframe of clustered proteins from neighborhoods...")
                mmseqs_clust = pn.prep_cluster_tsv(f"{args.out}combined_fastas_clust_res.tsv",logger)
                logger.debug(f"Total number of neighborhoods found: {len(set(mmseqs_clust['neighborhood_name']))}")
                

                if args.red_olp:
                    # reduce the number of neighborhoods that overalp in terms of location on the same chromosome
                    mmseqs_groups = list(mmseqs_clust.groupby(['gff', 'strand', 'seq_id'])) #cant groupby on its own with dask
                    mmseqs_groups = db.from_sequence(mmseqs_groups,npartitions=args.threads)
                    mmseqs_clust = db.map(pn.reduce_overlap,mmseqs_groups,window=args.olp_window)
                    del mmseqs_groups # save ram
                    mmseqs_clust = pd.concat(mmseqs_clust.compute())
                    logger.debug(f"Clustering df size after reducing number of overlapping neighborhoods: {mmseqs_clust.shape}")
                    logger.debug(f"Total number of neighborhoods after reducing number of overlapping neighborhoods: {len(set(mmseqs_clust['neighborhood_name']))}")
                del mmseqs_grp_db
                
                if (args.resume and len(removed_prot_gffs) == 0): # case if resume is called, and temp obj needs to recreated
                    try:
                        tmpfiles_backup = glob.glob(f"{args.out}tempdir_*/protgff_*")
                        for tmpf_path in tmpfiles_backup:
                            with open(tmpf_path,'r') as tmpf:
                                removed_prot_gffs.extend(tmpf.read().splitlines())
                        
                        if args.remove_temp:
                            shutil.rmtree(os.path.dirname(tmpfiles_backup)[0])
                    except Exception as e: # if the delete temp dir command was given
                        logger.error(e)

                mmseqs_clust = pn.map_vfcenters_to_vfdb_annot(mmseqs_clust,mmseqs_search,args.from_vfdb,removed_prot_gffs,logger)

                if args.remove_temp and tmpd != None:
                    shutil.rmtree(tmpd)

                if args.multi_vf_nn: 
                    # motivated by getting neighborhoods containing multiple VFs
                    mmseqs_clust = mmseqs_clust.compute()

                    if (not os.path.isfile(f"{args.out}multi_vf_search.tsv") and args.resume) or not (args.resume): # b/c mmseqs throws an error if the multi_vf_search db already exists
                        logger.debug("Searching additional queries in neighborhoods...")

                        subprocess.run(f"mmseqs search {args.out}queryDB {args.out}combined_fastas_db {args.out}multi_vf_search {args.out}tmp_search2 --min-seq-id {args.mvn_seq_id} --cov-mode 0 -c {args.mvn_cov} -v 2 --split-memory-limit {int(args.mem * (2/3))}G --threads {args.threads}  --alignment-mode 3 --start-sens 1 --sens-steps 3 -s 7"
                ,shell=True,check=True)
                        subprocess.run(["mmseqs", "convertalis", f"{args.out}queryDB", f"{args.out}combined_fastas_db", f"{args.out}multi_vf_search", f"{args.out}multi_vf_search.tsv", "--format-output", "query,target,evalue,pident,qcov,fident,alnlen,qheader,theader,tset,tsetid"] 
                ,check=True)
                    
                    multivf_mmseqs_grp_db,multi_vf_search = n.read_search_tsv(vfdb=args.from_vfdb,input_mmseqs=f"{args.out}multi_vf_search.tsv",threads=args.threads)

                    # get a list of neighborhoods that contain multiple hits from the query fasta (great for getting neighborhoods w/ muliple VFs)
                    mmseqs_clust,multi_query_nn = pn.select_multivf_neighborhoods(mmseqs_clust,loose_vf_search=multi_vf_search,logger=logger)
                    mmseqs_clust = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(multi_query_nn)]
                    mmseqs_clust = dd.from_pandas(mmseqs_clust,npartitions=args.threads)

                mmseqs_clust.to_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv",index=False,sep="\t")
                mmseqs_clust = mmseqs_clust.compute()
            
            elif len(glob.glob(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv"))>0 and args.resume:
                mmseqs_clust = dd.read_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust_*.tsv",sep="\t",dtype={'query': 'object',
                                                                                                                      'vf_category': 'object','vf_id': 'object',
                                                                                                                      'vf_name': 'object','vf_subcategory': 'object',
                                                                                                                      'vfdb_genus': 'object','vfdb_species': 'object'})
                mmseqs_clust = mmseqs_clust.compute() #reading with dask then computing is usually faster than read w/ pandas
                logger.info(f"Size of mmseqs cluster results after merge with search results: {mmseqs_clust.shape}")
            
            # make mmseqs_clust more memory efficient
            if 'vf_category' in mmseqs_clust.columns:
                mmseqs_clust = mmseqs_clust.astype({'query':'category','vf_category': 'category','vf_id': 'category',
                                                    'vf_name': 'category','vf_subcategory': 'category','vfdb_genus': 'category','vfdb_species': 'category'})
            else:
                mmseqs_clust = mmseqs_clust.astype({'query':'category'})
            
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            cluster_neighborhoods_by = "query"
            logger.debug("Creating groups of neighborhoods by their originial query")
            mmseqs_clust_nolink_groups = pn.get_query_neighborhood_groups(mmseqs_clust,cluster_neighborhoods_by)

            # run several analyses on neighborhoodss
            class_objs = {vf:pn.VF_neighborhoods(logger,mmseqs_clust_group,query=vf,dbscan_eps=0.15,dbscan_min=3)
                        for vf,mmseqs_clust_group in mmseqs_clust_nolink_groups}
            neighborhood_plt_df = pd.DataFrame.from_dict([class_objs[n].to_dict() for n in class_objs])
            neighborhood_plt_df = neighborhood_plt_df[neighborhood_plt_df['total_hits']>args.min_hits]
            neighborhood_plt_df['bubble_size'] = (100/(neighborhood_plt_df['noise']+1)).astype(int)
            del mmseqs_clust_nolink_groups

            if args.from_vfdb:
                # add vf info to neighborhood queries for glm embed color coding and other plotting
                mmseqs_clust_formerge = mmseqs_clust.drop_duplicates(subset=["query"])
                neighborhood_plt_df = pd.merge(neighborhood_plt_df,mmseqs_clust_formerge[['query','vf_name','vf_id','vf_subcategory','vf_category','vfdb_species','vfdb_genus']],on='query',how='left')
            neighborhood_plt_df.to_csv(f"{args.out}neighborhood_results_df.tsv",sep='\t',index=False,header=True)

            if args.glm:
                # take neighborhoods in class_objs and prep them to be fed into gLM for embeddings
                try:
                    glm_input_out = f"glm_inputs_{args.glm_cluster}_jaccard{str(args.glm_threshold)[1:]}/"
                    os.mkdir(args.out + glm_input_out) # should return an error if the path already exists, incase program finished in the middle of creating inputs, restart from here
                except FileExistsError as e: # if running w/ resume and glm_inputs directory is already made, clear it then make inputs from the beginning
                    logger.error(e)
                    logger.debug("Removing contents from glm inputs directory then recreating..")
                    shutil.rmtree(args.out + glm_input_out) # switched to shutil and os here b/c subprocess wasn't finding directory to remove for whatever reason
                    os.mkdir(args.out + glm_input_out)

                logger.debug("Grabbing cluster representatives...")
                if not os.path.isfile(f"{args.out}combined_fastas_clust_rep.fasta"): # save time if resuming
                    subprocess.run(f"mmseqs createsubdb {args.out}combined_fastas_clust {args.out}combined_fastas_db {args.out}combined_fastas_clust_rep",shell=True,check=True)
                    subprocess.run(f"mmseqs convert2fasta {args.out}combined_fastas_clust_rep {args.out}combined_fastas_clust_rep.fasta",shell=True,check=True)

                # reduce the number of similar neighborhoods by the amount of similar protein with a given threshold
                uniq_neighborhoods_d = {query:class_objs[query].get_neighborhood_names(args.glm_threshold,args.glm_cluster,logger) for query in class_objs}
                # get an idea of the number of filtered out neighborhoods that the glm will NOT see
                mmseqs_groups = mmseqs_clust.groupby('query')['neighborhood_name'].apply(list).to_dict()
                neighborhood_diff = {query:[len(mmseqs_groups[query]),len(uniq_neighborhoods_d[query])] for query in uniq_neighborhoods_d}
                pd.DataFrame.from_dict(neighborhood_diff,orient='index',columns=["Total_Neighborhoods","Total_Representative_Neighborhoods"]).to_csv(f"{args.out}Number_filtrd_Neighborhoods.tsv",sep="\t")
                
                # mmseqs cluster df can be really big, and can cause OOM issues when passed to so many threads. So I split up the df in "its" that should fit into mem
                logger.debug("Grabbing tsvs for glm input...")
                mmseqs_clust_mem = mmseqs_clust.memory_usage(deep=True).sum() / 10**8 # get mmseqs clust memory interms of GB 
                its = 1
                qs_for_glm = np.array(list(uniq_neighborhoods_d.keys()))
                while (mmseqs_clust_mem/its) * args.threads > args.mem:
                    its+=1
                qs_for_glm = np.array_split(qs_for_glm,its)

                # get the glm inputs for all neighborhoods, by iterating through a "chunk" number of neighborhoods
                logger.debug(f"Splitting mmseqs clustering df into {its} chunks...")
                for chunk in qs_for_glm:
                    neighborhoods_to_subset_for = sum([mmseqs_groups[q] for q in chunk],[])
                    mmseqs_clust_sub = mmseqs_clust[mmseqs_clust['neighborhood_name'].isin(neighborhoods_to_subset_for)]
                    db.map(glm.get_glm_input,query=db.from_sequence(chunk,npartitions=args.threads),
                        uniq_neighborhoods_d=uniq_neighborhoods_d,neighborhood_res=neighborhood_plt_df,mmseqs_clust=mmseqs_clust_sub,glm_input_dir=glm_input_out,vfdb=args.from_vfdb,logger=logger,args=args).persist()
                    
        elif args.plt_from_saved:
            neighborhood_plt_df = pd.read_csv(args.plt_from_saved,sep='\t')
        if args.plot:
            pn.plt_neighborhoods(neighborhood_plt_df,args.out,vfdb=args.from_vfdb)
            pn.plt_hist_neighborh_clusts(neighborhood_plt_df,args.out)
            pn.plt_regline_scatter(neighborhood_plt_df,args.out)
            pn.plt_box_entropy(neighborhood_plt_df,out=args.out,vfdb=args.from_vfdb)

    
    if args.subcommand == "compare_neighborhoods":
        logger = get_logger(args.subcommand,args.out)
        logger.debug("Comparing neighbors...")
        dirs_l = check_dirs(args.out)
        args.out = dirs_l[0]
        neighborhood1,neighborhood2 = pd.read_csv(args.neighborhood1,sep='\t'),pd.read_csv(args.neighborhood2,sep='\t')
        c.compare_neighborhood_entropy(neighborhood1,neighborhood2,label1=args.name1,label2=args.name2,out=args.out)
        c.compare_uniqhits_trends(neighborhood1,neighborhood2,label1=args.name1,label2=args.name2,out=args.out,write_table=True)


    if args.subcommand == "chop-genome":
        logger = get_logger(args.subcommand,args.out)
        logger.debug("Chopping up some genomes...")
        dirs_l = check_dirs(args.genomes,args.out)
        args.genomes,args.out = dirs_l[0],dirs_l[1]

        genome_queries = [genome.split('/')[-1].split('.gff')[0] for genome in glob.glob(args.genomes + '*') if '.gff' in genome]
        logger.debug(f"All genome names to get neighborhoods from: {genome_queries}")

        neighborhood_db = db.map(n.get_neigborhood,logger,args,genome_query = db.from_sequence(genome_queries,npartitions=args.threads))
        neighborhood_db = neighborhood_db.flatten()
        n.run_fasta_from_neighborhood(logger,args,dir_for_fasta=args.genomes,neighborhood=neighborhood_db,
                                      out_folder=args.out,threads=args.threads)
        
        # grab all the proteins found from all neighborhoods, to then send to a dataframe with neighborhood id info
        combinedfastas = glob.glob(f"{args.out}combined_fasta_partition*")
        singular_combinedfasta = f"{args.out}combined.fasta"
        with open(singular_combinedfasta,"w") as outfile:
            for fasta in combinedfastas:
                records = glm.SeqIO.parse(fasta, "fasta")
                glm.SeqIO.write(records, outfile, "fasta")
        logger.debug(f"Combined fastas into {singular_combinedfasta}")

        # i think its fasta to open the singular combined fasta and loop once, then looping through the records of each partition, didnt test this tho!
        allids = [rec.id for rec in glm.SeqIO.parse(singular_combinedfasta,"fasta")]

        # simulate an mmseqs clustering df output without actually clustering proteins, so that written functions work
        mmseqs_clust = pd.DataFrame([allids,allids]).T
        mmseqs_clust.columns = ['rep','locus_tag'] 
        mmseqs_clust = pn.prep_cluster_tsv(mmseqs_clust,logger)
        logger.debug(f"Total number of queries: {len(mmseqs_clust['query'])}")
        logger.debug(f"Total number of neighborhoods: {len(mmseqs_clust['neighborhood_name'])}")

        # b/c we didnt do an initial search all VF centers are the queries for glm input purposes
        mmseqs_clust['query'] = mmseqs_clust['VF_center'].copy() 
        subprocess.run(f"mkdir {args.out}clust_res_in_neighborhoods",shell=True,check=True)
        mmseqs_clust.to_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust.tsv",sep="\t",index=False)

        glm_input_out = "glm_inputs"
        subprocess.run(f"mkdir {args.out}{glm_input_out}/",shell=True) # should return an error if the path already exists, don't want to make duplicates

        # create tsvs for glm inputs, one tsv per protein
        db.map(glm.get_glm_input,query=db.from_sequence(list(set(mmseqs_clust['query'])),npartitions=args.threads),
               mmseqs_clust=mmseqs_clust,combinedfasta=singular_combinedfasta,glm_input_dir=glm_input_out,logger=logger,args=args).compute()
    
    if args.subcommand == 'predictvf':
        logger = get_logger(args.subcommand,args.out)
        logger.debug("Chopping up some genomes...")
        dirs_l = check_dirs(args.genomes,args.out,args.include_structhits)
        args.genomes,args.out,args.include_structhits = dirs_l[0],dirs_l[1],dirs_l[2]
        pkl_objs = l.load_pickle()
        
        if not args.glm_embeds:
            # if (args.resume and not os.path.isfile(f'{args.out}clust_res_in_neighborhoods/mmseqs_clust.tsv')) or (not args.resume):
            logger.debug(f"Pulling neighborhoods from {len(glob.glob(args.genomes))/2} pairs of proteomes + gffs...")
            mmseqs_clust,singular_combinedfasta = nc.pull_neighborhoodsdf(args,logger)

            # if (args.resume and os.path.isfile(f'{args.out}clust_res_in_neighborhoods/mmseqs_clust.tsv')) or (not args.resume):
            glm_input_out = "glm_inputs"
            try:
                os.mkdir(args.out + glm_input_out) # should return an error if the path already exists, incase program finished in the middle of creating inputs, restart from here
            except FileExistsError as e: # if running w/ resume and glm_inputs directory is already made, clear it then make inputs from the beginning
                logger.error(e)
                logger.debug("Removing contents from glm inputs directory then recreating..")
                shutil.rmtree(args.out + glm_input_out) # switched to shutil and os here b/c subprocess wasn't finding directory to remove for whatever reason
                os.mkdir(args.out + glm_input_out)

                # create tsvs for glm inputs, one tsv per protein
                logger.debug(f"Transforming collected {len(set(mmseqs_clust['query']))} proteins for input into the gLM...")
                db.map(glm.get_glm_input,query=db.from_sequence(list(set(mmseqs_clust['query'])),npartitions=args.threads),
                        mmseqs_clust=mmseqs_clust,combinedfasta=singular_combinedfasta,glm_input_dir=glm_input_out,logger=logger,args=args).compute()
            
            glm_ouputs_out = "glm_outputs"
            subprocess.run(f"mkdir {args.out}{glm_ouputs_out}/",shell=True)
            logger.debug("Computing pLM embeddings...")
            nc.get_plm_embeds(args.out+glm_input_out, args.out+glm_ouputs_out)

            tsvs_to_getembeds = glob.glob(args.out+glm_input_out+"/*.tsv")
            computed_embed_names = [f.split('/')[-1] for f in glob.glob(args.out+glm_ouputs_out+"/*")] #only keep the query name from split for easier identification
            files = [file.split('.tsv')[0] for file in tsvs_to_getembeds if file.split('/')[-1].split('.tsv')[0] not in computed_embed_names] # make sure embed hasnt been computed already
            
            logger.debug(f"Computing gLM embeddings for: {len(files)} proteins")

            files_db = db.from_sequence(files,npartitions=args.threads) #I dont think npartitions matters too much here, number of parllelizes calls will be equal to # of gpus available
            
            res_name = db.map(nc.create_glm_embeds,files_db,args.out+glm_ouputs_out,norm_factors=pkl_objs['norm.pkl'],PCA_LABEL=pkl_objs['pca.pkl'],ngpus=args.ngpus)
            res_name = res_name.compute()

            mmseqs_clust = pd.read_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust.tsv",sep='\t')
            glm_res_d_vals_predf =  cu.unpack_embeddings(args.out+glm_ouputs_out,args.out+glm_input_out,mmseqs_clust)
            embedding_df_merge = cu.get_glm_embeddf(glm_res_d_vals_predf)
            embedding_df_merge.to_csv(f"{args.out}glm_embeds.tsv",sep="\t",index=False)
        
        if args.glm_embeds:
            embedding_df_merge = pd.read_csv(args.glm_embeds,sep="\t")
            mmseqs_clust = dd.read_csv(f'{args.include_structhits}clust_res_in_neighborhoods/mmseqs_clust*',sep="\t",dtype={'query': 'object'}).compute()

        lb = pkl_objs['labelbinarizer_vfcategories.obj']
        models_dict = l.load_clf_models()
        nn_preds_res = nc.get_embed_preds(embedding_df_merge,models_dict['nn_clf'],lb=lb,args=args)
        nn_preds_res.to_csv(f"{args.out}neighborhood_based_predictions.tsv",sep="\t",index=False)

        # pull together structure search results
        logger.debug("Foldseek search query proteins against VF and NS database...")

        struct_search_raw = sc.foldseek_search(args)
        tsvs_d = l.load_vf_functional_mappers()
        vfid_mapping,vfquery_vfid = tsvs_d['VFID_mapping_specified.tsv'],tsvs_d['vfquery_to_id_tocat.tsv']
        struct_search = sc.format_search([struct_search_raw],meta = ['q_vn'],vfquery_vfid=vfquery_vfid,vfmap_df=vfid_mapping)
        struct_search = sc.format_searchlabels(struct_search)
        struct_search.to_csv(f"{args.out}foldseek_search_labelsmapped.tsv",sep="\t",index=False)

        logger.debug("Collecting best structural similarity for each functional group if exists...")
        struct_search_queries = set(struct_search['query'])
        logger.info(f"Structure-based search results for {len(set(nn_preds_res['query']) - struct_search_queries)} queries could not be found...")
        queries_db = db.from_sequence(struct_search_queries,npartitions = args.threads)
        pred_raw = db.map(sc.alltopNhits_probs_threadable,queries_db,df=struct_search[['query','target','qtmscore','tvf_category']],score_metric="qtmscore",lb=lb)
        pred_raw = pred_raw.compute()
        struct_preds_res = sc.format_strucpreds(pred_raw=pred_raw,lb=lb)
        struct_preds_res.to_csv(f"{args.out}structure_based_predictions.tsv",sep="\t",index=False)

        logger.debug("Integrating neighborhood and structure based predictions")
        nn_struct_preds = pd.merge(nn_preds_res,struct_preds_res,on='query',how='left')
        
        nn_struct_preds.fillna(0.0,inplace=True) # some queries don't show up in structure search b/c no hits. Which is why there are more neighborhoods than struct hits
        integrated_preds = clf.meta_classifier(nn_struct_preds=nn_struct_preds,model=models_dict['int_clf'],lb=lb)

        if args.include_structhits:
            mmseqs_clust['true_lc'] = mmseqs_clust['locus_tag'].str.split('!!!').str[0] # b/c these are similar to names used in foldseek search

            # define cutoff for structural hits
            struct_search_hits = struct_search[(struct_search['tvf_category']!='non_vf') & 
                                   ((struct_search['qtmscore']>args.qtmcutoff) | ((struct_search['qcov']>args.fs_qcovcutoff) & (struct_search['lddt']>args.lddtcutoff)))]
            
            # get a dictionary for the number of VF structural hits in a neighborhood
            mmseqsclust_structhits = mmseqs_clust[mmseqs_clust['true_lc'].isin(struct_search_hits['query'])]

            hits_per_nn = mmseqsclust_structhits.groupby('neighborhood_name')['true_lc'].apply(len).to_dict() # get the number of structural hits in each neighborhood
            integrated_preds['nn_struct_hits'] = integrated_preds['neighborhood_name'].map(hits_per_nn) # map hits to neighborhood names
            integrated_preds['nn_struct_hits'] = integrated_preds['nn_struct_hits'].fillna(0) # fill neighborhoods with no hits with 0

        integrated_preds.to_csv(f"{args.out}integrated_predictions.tsv",sep="\t",index=False)


    if args.subcommand == "compute_umap": # gotta change the functions used for this
        logger = get_logger(args.subcommand,args.out)
        logger.debug("Computing umap from gLM...")

        # make sure directories have a slash at the end
        if len(args.out) > 1:
            dirs_l = check_dirs(args.neighborhood_run,args.glm_out,args.glm_in,args.out)
            neighborhood_dir,glm_out,glm_in,args.out = dirs_l[0],dirs_l[1],dirs_l[2],dirs_l[3]
            if not os.path.exists(args.out):
                os.mkdir(args.out)
        else:
            dirs_l = check_dirs(args.neighborhood_run,args.glm_out,args.glm_in)
            neighborhood_dir,glm_out,glm_in = dirs_l[0],dirs_l[1],dirs_l[2]

        mmseqs_clust = dd.read_csv(f"{neighborhood_dir}clust_res_in_neighborhoods/mmseqs_clust*.tsv",sep="\t",dtype={'query': 'object'})
        mmseqs_clust = mmseqs_clust.compute()

        logger.debug(f"Grabbing gLM from {glm_out} \nGiven these inputs: {glm_in}")
        umapper,embedding_df_merge = args.umap_obj,args.embedding_df
        # for whatever reason the below doesn't work on the first try, have to rerun with the umap_obj and embedding_df arguments given
        if (not args.umap_obj) and (not args.embedding_df):
            glm_res_d_vals_predf =  cu.unpack_embeddings(glm_out,glm_in,mmseqs_clust)
            embedding_df_merge = cu.get_glm_embeddf(glm_res_d_vals_predf)
            embedding_df_merge.to_csv(f"{args.out}glm_embeds.tsv",sep="\t",index=False)
            umapper = cu.get_umapdf(embedding_df_merge)
            handle = open(f"{args.out}umapper.obj","wb")
            pkl.dump(umapper,handle)
            handle.close()
        else:
            handle = open(args.umap_obj,"rb")
            umapper = pkl.load(handle)
            handle.close()
            embedding_df_merge = pd.read_csv(args.embedding_df,sep="\t")
        
        if args.plt_umap:
            cu.plt_baby(umapper,embedding_df_merge,plt_name=args.umap_name,outdir=args.out,
                        legend=args.legend,width=args.width,label=args.label)

    logger.debug(f"Done! Took --- %s seconds --- to complete" % (time.time() - start_time))
    return

def run():
    # if __name__ == "__main__":
    parser = get_parser()
    workflow(parser)
    quit()

