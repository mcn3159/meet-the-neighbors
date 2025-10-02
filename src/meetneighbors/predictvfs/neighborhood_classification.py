import argparse
import sys
import tempfile
import subprocess
import pandas as pd
import numpy as np
import glob
import pickle as pkl
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio import SeqIO
import importlib.resources
import pdb
import os

import dask
import dask.bag as db
import dask.dataframe as dd
import torch
import torch.nn.functional as F
from transformers import RobertaConfig


import meetneighbors.neighbors_frm_mmseqs as n
import meetneighbors.glm_input_frm_neighbors as glm
import meetneighbors.plot_map_neighborhood_res as pn
import meetneighbors.predictvfs.clf_models as clf

import meetneighbors.predictvfs.glm.plm_embed as plm
import meetneighbors.predictvfs.glm.batch_data as bd
import meetneighbors.predictvfs.glm.glm_embed as glm_e
from meetneighbors.predictvfs.glm.gLM import *

import meetneighbors.ring_mmseqs as mm


def pull_neighborhoodsdf(args,tmpd,logger):

    if args.resume:
        if args.cluster:
            file_to_check = f'{args.out}combined_fastas_clust_res.tsv'
        else:
            file_to_check = f"{args.out}combined.fasta"
        if args.query_fasta: # this resume check doesnt work for prot_genome_pairs and maybe chop-genomes mode. Adjust soon, tired now
            mmseqs_grp_db,mmseqs_search = n.read_search_tsv(input_mmseqs=f"{args.out}vfs_in_genomes.tsv",threads=args.threads)
        
    if (args.resume and not os.path.isfile(file_to_check)) or (not args.resume):
        if args.query_fasta:
            if args.prot_genome_pairs:
                mmseqs_search = pd.read_csv(args.prot_genome_pairs,sep="\t",names=['target','tset']) # this is a pseudo mmseqs search, b/c it will match requirements for future functions
                mmseqs_search['query'] = mmseqs_search['target']
                mmseqs_grp_db = list(mmseqs_search.groupby('tset'))
                mmseqs_grp_db = db.from_sequence(mmseqs_grp_db,npartitions=args.threads)
            else:
                mmseqs_grp_db,mmseqs_search = n.read_search_tsv(input_mmseqs=f"{args.out}vfs_in_genomes.tsv",threads=args.threads)
                logger.info(f"Number of query proteins found in genomes: {len(set(mmseqs_search['query']))}")
            neighborhood_db = db.map(n.get_neigborhood,logger,args,tmpd, mmseqs_groups = mmseqs_grp_db,head_on=args.head_on,intergenic_cutoff=args.intergenic) # might want to fix the potential issue of neighborhoods getting removed if they don't meet minimum criteria
            logger.debug("Saving protiens found in neighborhoods..")
            neighborhood_db = neighborhood_db.flatten()
            n.run_fasta_from_neighborhood(logger,args,dir_for_fasta=args.genomes,neighborhood=neighborhood_db,
                                            out_folder=args.out,threads=args.threads) # need to call this function in a smarter way. Basically I'm opening and closing a new fasta for neighborhood, which is probably a major slow down.
            # maybe instead of the above two lines I can...
            # 1) pd.concat the neighborhood dataframes in neighborhooddb
            # 2) groupby gff name then send the groups to a list, then convert to dask bag
            # 3) Apply the run_fasta_from_neighborhood() function to each group (would require me to modify run_fasta_from_neighborhood()

        elif args.genomes or args.genome_tsv:
            logger.debug("Chopping up some genomes...")
            if args.genome_tsv: # use if user provides a tsv with paths to genome and protein files, columns must be genome_name, protein path, and gff path
                args.genome_tsv = pd.read_csv(args.genome_tsv,sep="\t",names=['genome','protein','gff'])
                firstrow_check = args.genome_tsv.iloc[0]
                file_endings = {'.gff':'gff',
                                '.faa':'protein',
                                '.fasta':'protein'}
                find_correct_colnames = {}

                for i,colname in enumerate(firstrow_check):
                    for ending,v in file_endings.items():
                        if ending in colname:
                            find_correct_colnames[i] = v
                            break
                        else:
                            find_correct_colnames[i] = 'genome'
                args.genome_tsv.columns = [find_correct_colnames[i] for i in range(3)] # should be up to 3 columns

                genome_queries = list(args.genome_tsv['genome'])
                genome_query_db = db.from_sequence(genome_queries,npartitions=args.threads)
                neighborhood_db = db.map(n.get_neigborhood,logger,args,tmpd, genome_query = genome_query_db,head_on=args.head_on,intergenic_cutoff=args.intergenic)
            else:
                genome_queries = [genome.split('/')[-1].split('.gff')[0] for genome in glob.glob(args.genomes + '*.gff')]
                genome_query_db = db.from_sequence(genome_queries,npartitions=args.threads)
                neighborhood_db = db.map(n.get_neigborhood,logger,args,tmpd, genome_query = genome_query_db,head_on=args.head_on,intergenic_cutoff=args.intergenic)
                
            neighborhood_db = neighborhood_db.flatten().compute()
            neighborhood_db = pd.concat(neighborhood_db)            
            neighborhood_db.rename(columns={'protein_id':'locus_tag'},inplace=True)
        else:
            raise Exception("Inputs arguments must include --genomes and --query_fasta, or --genome_tsv, or --prot_genome_pairs")
    
    if args.cluster: # need the mmseqs_search object from a query fasta from this work
        logger.debug("Clustering proteins found in all neighborhoods...")
        mm.mmseqs_cluster(args)
        singular_combinedfasta = f"{args.out}combined_fastas_clust_rep.fasta" # to match inputs into glm_input_frm_neighbors.py
        mmseqs_clust = f"{args.out}combined_fastas_clust_res.tsv"
        
    else:
        # grab all the proteins found from all neighborhoods, to then send to a dataframe with neighborhood id info
        if isinstance(args.genome_tsv,pd.DataFrame):
            combinedfastas = list(args.genome_tsv['protein'])
            genome_query=True
        else:
            combinedfastas = glob.glob(f"{args.out}combined_fasta_partition*")
            genome_query=False
        
        singular_combinedfasta = f"{args.out}combined.fasta"
        with open(singular_combinedfasta,"w") as outfile:
            for fasta in combinedfastas:
                records = glm.SeqIO.parse(fasta, "fasta")
                glm.SeqIO.write(records, outfile, "fasta")

        # i think its faster to open the singular combined fasta and loop once, then looping through the records of each partition, didnt test this tho!
        allids = [rec.id for rec in glm.SeqIO.parse(singular_combinedfasta,"fasta")]
        # simulate an mmseqs clustering df output without actually clustering proteins, so that written functions work
        mmseqs_clust = pd.DataFrame([allids,allids]).T
        mmseqs_clust.columns = ['rep','locus_tag'] 
    
    # b/c we didnt do an initial search all VF centers are the queries for glm input purposes. This will end up creating a lot of glm_inputs
    if args.query_fasta:
        mmseqs_clust = pn.prep_cluster_tsv(mmseqs_clust,logger)
        mmseqs_clust = pn.map_vfcenters_to_vfdb_annot(mmseqs_clust,mmseqs_search,vfdb=None,removed_neighborhoods=None,logger=logger) # need to fix removed_neighborhoods later
        mmseqs_clust = mmseqs_clust.compute()
        # mmseqs_clust['query'] = mmseqs_clust['VF_center'].copy()
    else:
        mmseqs_clust = pn.prep_cluster_tsv(mmseqs_clust,logger,genome_query=genome_query)
        mmseqs_clust = pd.merge(mmseqs_clust,neighborhood_db[['locus_tag','start','strand','neighborhood_name']],on='locus_tag',how='right')
        mmseqs_clust['locus_tag'] = mmseqs_clust['locus_tag'] + '!!!' + mmseqs_clust['neighborhood_name']
        mmseqs_clust['query'] = mmseqs_clust['neighborhood_name'].str.split('!!!').str[1]
    
    nn_hash = None
    if args.memory_optimize:
        logger.debug("Save then remove duplicate neighborhoods. Map neighborhoods back via embeddings")
        logger.debug(f"Total number of neighborhoods pre-hash: {len(set(mmseqs_clust['neighborhood_name']))}")
        
        mmseqs_clust, nn_hash = pn.hash_neighborhoods(mmseqs_clust,args)
        pkl.dump(nn_hash,open(f'{args.out}nnhash_mapping.obj','wb'))

    logger.debug(f"Total number of neighborhoods post-hash: {len(set(mmseqs_clust['neighborhood_name']))}")
    try: # incase resume was called
        subprocess.run(f"mkdir {args.out}clust_res_in_neighborhoods",shell=True,check=True) 
    except:
        subprocess.run(f"rm -r {args.out}clust_res_in_neighborhoods",shell=True,check=True)
        subprocess.run(f"mkdir {args.out}clust_res_in_neighborhoods",shell=True,check=True)
    mmseqs_clust.to_csv(f"{args.out}clust_res_in_neighborhoods/mmseqs_clust.tsv",sep="\t",index=False)

    return mmseqs_clust, singular_combinedfasta, nn_hash

def checkXle(fasta_recs):
    # pLM used for gLM doesn't have a token for J amino acids (or Xle) or replace w/ the most similar, Leucine
    for rec in fasta_recs:
        prot = rec.seq
        if "J" in prot:
            mutable_seq = MutableSeq(prot)
            mutable_seq = mutable_seq.replace("J","L")
            new_seq = Seq(mutable_seq)
            rec.seq = new_seq
    #assert len([rec for rec in fasta if "J" in rec.seq]) ==0, "Records with weird J amino acid still in fasta"
    return fasta_recs

def get_plm_embeds(glm_inputs_path,glm_outputs_path):
    subprocess.run(f"cat {glm_inputs_path}/*.fasta > {glm_outputs_path}/all_glm_input_prots.fasta",shell=True,check=True)
    unique_records = {record.id: record for record in SeqIO.parse(f"{glm_outputs_path}/all_glm_input_prots.fasta","fasta")} #some neighborhoods may have the same proteins, which will be duplicated in og_fasta
    fasta_recs = checkXle(unique_records.values())
    with open(f"{glm_outputs_path}/all_glm_input_prots_reps.fasta","w") as handle:
        SeqIO.write(fasta_recs,handle,"fasta")

    sequence_representations = plm.run_plm(f"{glm_outputs_path}/all_glm_input_prots_reps.fasta")
    try:
        fi = open(f"{glm_outputs_path}/all_glm_input_prots_reps.esm.embs.pkl", "wb")
        pkl.dump(sequence_representations,fi)
        fi.close()
    except pkl.PicklingError:
        print("Ran into an error when pickling plm embeddings, opening console to debug..")
        pdb.set_trace()
    return

def create_glm_embeds(f,glm_outputs_path,norm_factors,PCA_LABEL,ngpus,bs):
    # need to incorporate pkl files from glm (norm and pca.pkl)
    # batch_data_path = importlib.resources.path("meetneighbors.predictvfs.glm","batch_data.py")
    glm_model = importlib.resources.path("meetneighbors.predictvfs.glm.model","glm.bin")
    res_name = f.split('/')[-1] 
    subprocess.run(f"mkdir '{glm_outputs_path}/{res_name}'",shell=True,check=True) #multiple queries with the same name?
    
    batched_dir = f'{glm_outputs_path}/{res_name}/batched'
    subprocess.run(f"mkdir {batched_dir}",shell=True,check=True)

    # subprocess.run(f"python {batch_data_path} {glm_outputs_path}/all_glm_input_prots_reps.esm.embs.pkl '{f}'.tsv '{glm_outputs_path}/{res_name}/batched'",shell=True,check=True)
    bd.run_batcher(f"{glm_outputs_path}/all_glm_input_prots_reps.esm.embs.pkl",f"{f}.tsv",norm_factors,PCA_LABEL,f"{glm_outputs_path}/{res_name}/batched")
    # subprocess.run(f"python {glm_embed_path} -d '{glm_outputs_path}/{res_name}/batched' -m {glm_model} -b 100 -o '{glm_outputs_path}/{res_name}/results'",shell=True,check=True)
    num_pred = 4
    max_seq_length = 30 
    num_attention_heads = 10
    num_hidden_layers= 19
    pos_emb = "relative_key_query"
    pred_probs = True
    HIDDEN_SIZE = 1280
    EMB_DIM = 1281
    NUM_PC_LABEL = 100
     # populate config 
    config = RobertaConfig(
        max_position_embedding = max_seq_length,
        hidden_size = HIDDEN_SIZE,
        num_attention_heads = num_attention_heads,
        type_vocab_size = 1,
        tie_word_embeddings = False,
        num_hidden_layers = num_hidden_layers,
        num_pc = NUM_PC_LABEL, 
        num_pred = num_pred,
        predict_probs = pred_probs,
        emb_dim = EMB_DIM,
        output_attentions=True,
        output_hidden_states=True,
        position_embedding_type = pos_emb,
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device,flush=True)
    model = gLM(config)
    model.load_state_dict(torch.load(glm_model, map_location=device),strict=False)
    glm_e.run_glm_embeds(model,pkg_data_dir=batched_dir,glm_embed_output_path=f'{glm_outputs_path}/{res_name}/results',device=device,ngpus=ngpus,batch_size=bs)
    return f

def get_embed_preds(embeds,model_weights,lb,args): # might want to put lb into the argparse
    input_dim,num_classes = 1280,len(lb.classes_)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = clf.FullyConnectedNN(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_weights))
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(embeds.iloc[:,1:1+1280].values.astype(np.float32)).to(device))
        ecc_predictions = pd.DataFrame(F.softmax(outputs,dim=1).detach().cpu().numpy())
            
    ecc_predictions.columns = [cat for cat in lb.classes_]

    ecc_predictionsdf = pd.concat([embeds['query'],embeds['neighborhood_name'],ecc_predictions],axis=1)

    # So the annotations rarely ever mapped with the code commented out below. Don't think it's worth fixing, was just nice to have annotations in results
    # map annotations to predictionsdf
    # prot_annots = {}
    # prot_faas = [proteome for proteome in glob.glob(args.genomes+'*') if '.gff' not in proteome]
    # for fasta in prot_faas:
    #     prot_annots.update({rec.id.split('|')[-1]:' '.join(rec.description.split(' ')[1:]) for rec in SeqIO.parse(fasta,"fasta")})
    
    # neighborhoods couldn't be extracted for some queries b/c it didn't fit the minimum neighborhood definition requirements
    # logger.debug("Number of queries with no neighborhoods:",len(prot_annots)-len(set(embeds['query'])))
    
    # ecc_predictionsdf['seq_annotations'] = ecc_predictionsdf['query'].map(prot_annots)
    
    new_col_order = ['query', 'neighborhood_name'] + list(lb.classes_)

    ecc_predictionsdf = ecc_predictionsdf[new_col_order]
    
    return ecc_predictionsdf