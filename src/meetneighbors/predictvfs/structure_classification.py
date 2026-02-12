# for structure
import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
from statistics import geometric_mean
import pickle as pkl
from sklearn.preprocessing import StandardScaler # type: ignore
import time
import tqdm # type: ignore
import importlib.resources
import shutil

def foldseek_search(args,logger):
    # search query's foldseek db against targetdb
    
    foldseek_search_out = "foldseek_search_output" # directory for the foldseek search results
    try:
        os.mkdir(f"{args.out}{foldseek_search_out}/")
    except FileExistsError as e:
        logger.error(e)
        logger.debug("Removing contents from Foldseeek search then recreating..")
        shutil.rmtree(args.out + foldseek_search_out) # switched to shutil and os here b/c subprocess wasn't finding directory to remove for whatever reason
        os.mkdir(args.out + foldseek_search_out)

    concat_db = importlib.resources.path("meetneighbors.predictvfs.data.vf_ns_foldseekdb","vfnsconcatdb")
    subprocess.run(f"foldseek search {args.foldseek_structs} {concat_db} {args.out}{foldseek_search_out}/foldseek_search_res {args.out}{foldseek_search_out}/foldseek_tmp --exhaustive-search  --threads {args.threads} -a"
        ,shell=True,check=True)
    
    subprocess.run(["foldseek", "convertalis", args.foldseek_structs, concat_db, f"{args.out}{foldseek_search_out}/foldseek_search_res", f"{args.out}{foldseek_search_out}/foldseek_search_res.tsv","--threads", str(args.threads), "--format-output", "query,qheader,target,theader,prob,qlen,alnlen,qstart,pident,qcov,alntmscore,qtmscore,ttmscore,lddt,bits,evalue"])

    struct_search_raw = pd.read_csv(f"{args.out}{foldseek_search_out}/foldseek_search_res.tsv",sep="\t", names = ["query","qheader","target","theader","prob","qlen","alnlen","qstart","pident","qcov","alntmscore","qtmscore","ttmscore","lddt","bits","evalue"])
    return struct_search_raw


def vfid_query_mapper(vfquery_id_path,vfid_mapping_path):
    # create objects that allows for mapping of functional labels to VF database

    vfid_mapping = pd.read_csv(vfid_mapping_path,sep="\t") # load in mapping file of vfid to subcategory
    vfid_mapping = vfid_mapping.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # remove extra spaces in file

    vfid_vfquery_df = pd.read_csv(vfquery_id_path,sep="\t")
    vfquery_vfid = dict(zip(vfid_vfquery_df['query'],vfid_vfquery_df['VFID']))

    return vfid_mapping,vfquery_vfid


def format_search(search_res,meta,vfquery_vfid,vfmap_df):
    # Combine multiple foldseek searches or format a single search into one df by adding function labels info to the df
    
    assert isinstance(search_res,list), "Input object needs to be a list"
    assert len(meta) == len(search_res), "Need meta data like ['q_v'] to be the same lengthe as search_res"
    meta_options = ['q_vn','q_v', 'n_v', 'v_n', 'v_v', 'q_n', 'n_n']
    
    for df in search_res:
        df['query'] = df['query'].str.split('.pdb').str[0]
        df['target'] = df['target'].str.split('.pdb').str[0]
        df.drop(columns = ['qheader','theader'],inplace=True)
        
    # copy is here to change column names in peace
    vfmap_df_copy = vfmap_df.copy()
    
    for i,m in enumerate(meta):
        df = search_res[i].copy()
        
        if m == 'q_vn': # search was setup a queries against vfs and ns
            df['tVFID'] = df['target'].map(vfquery_vfid)
            vfmap_df_copy.rename(columns={'VFID':'tVFID'},inplace=True)
            df = pd.merge(df,vfmap_df_copy[['tVFID','vf_category0_subcat0','vf_category0_subcat1']],on='tVFID',how='left')
            df.fillna('non_vf',inplace=True)
            df.rename(columns={'vf_category0_subcat0':'tvf_category','vf_category0_subcat1':'tvf_subcategory'},inplace=True)
        
        if m == 'q_v':
            df['tVFID'] = df['target'].map(vfquery_vfid)
            vfmap_df.rename(columns={'VFID':'tVFID'},inplace=True)
            df = pd.merge(df,vfmap_df[['tVFID','vf_category0_subcat0','vf_category0_subcat1']],on='tVFID')
            df.rename(columns={'vf_category0_subcat0':'tvf_category','vf_category0_subcat1':'tvf_subcategory'},inplace=True)
            vfmap_df = vfmap_df_copy.copy()
            search_res[i] = df.copy()
            
        if m == 'n_v':
            df['tVFID'] = df['target'].map(vfquery_vfid)
            vfmap_df.rename(columns={'VFID':'tVFID'},inplace=True)
            df = pd.merge(df,vfmap_df[['tVFID','vf_category0_subcat0','vf_category0_subcat1']],on='tVFID')
            df.rename(columns={'vf_category0_subcat0':'tvf_category','vf_category0_subcat1':'tvf_subcategory'},inplace=True)
            df[['qVFID','qvf_category','qvf_subcategory']] = 'non_vf'
            vfmap_df = vfmap_df_copy.copy()
            search_res[i] = df.copy()
            
        if m == 'v_n':
            df['qVFID'] = df['query'].map(vfquery_vfid)
            df = pd.merge(df,vfmap_df[['qVFID','vf_category0_subcat0','vf_category0_subcat1']],on='qVFID')
            df.rename(columns={'vf_category0_subcat0':'qvf_category','vf_category0_subcat1':'qvf_subcategory'},inplace=True)
            df[['tVFID','tvf_category','tvf_subcategory']] = 'non_vf'
            vfmap_df = vfmap_df_copy.copy()
            search_res[i] = df.copy()
        
        if m == 'v_v':
            df['tVFID'] = df['target'].map(vfquery_vfid)
            df['qVFID'] = df['query'].map(vfquery_vfid)
            vfmap_df.rename(columns={'VFID':'tVFID'},inplace=True)
            df = pd.merge(df,vfmap_df[['tVFID','vf_category0_subcat0','vf_category0_subcat1']],on='tVFID')
            df.rename(columns={'vf_category0_subcat0':'tvf_category','vf_category0_subcat1':'tvf_subcategory'},inplace=True)
            vfmap_df.rename(columns={'tVFID':'qVFID'},inplace=True)
            df = pd.merge(search_res,vfmap_df[['qVFID','vf_category0_subcat0','vf_category0_subcat1']],on='qVFID')
            df.rename(columns={'vf_category0_subcat0':'qvf_category','vf_category0_subcat1':'qvf_subcategory'},inplace=True)
            vfmap_df = vfmap_df_copy.copy()
            search_res[i] = df.copy()
            
        if m == 'q_n':
            df[['tVFID','tvf_category','tvf_subcategory']] = 'non_vf'
            search_res[i] = df.copy()
        
        if m == 'n_n':
            df[['qVFID','qvf_category','qvf_subcategory','tVFID','tvf_category','tvf_subcategory']] = 'non_vf'
            search_res[i] = df.copy()
            
        elif m not in meta_options:
            raise ValueError(f"Meta param: {m} not an appropriate format, options are {meta_options}")
            
        search_res[i] = df
    
    search_res = pd.concat(search_res)
    # search_res['mean_score'] = search_res['bits'] * geometric_mean([search_res['qtmscore'], search_res['lddt']])
    search_res['geo_mean'] = search_res.apply(lambda row: geometric_mean([row['lddt'], row['alntmscore']]), axis=1)
    search_res['mean_score'] = search_res['bits'] * search_res['geo_mean']
    return search_res

def format_searchlabels(search_res):
    # add additional functional labels to VF hits that don't have a subcat functional label, ignore categories that we didnt train

    classes_to_skip = ['Others','Antimicrobial activity/Competitive advantage','Stress survival','Biofilm']
    search_res = search_res[~search_res['tvf_category'].isin(classes_to_skip)]

    # replacing nonvf subcat labels for vf categories with "unknown_vfsubcat", nonvfs labels for vfs came from format_search()
    search_res.loc[(search_res['tvf_category']=='Exotoxin') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory'] = search_res.loc[(search_res['tvf_category']=='Exotoxin') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory']='unknown_Exotoxin_subcat'
    search_res.loc[(search_res['tvf_category']=='Immune modulation') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory'] = search_res.loc[(search_res['tvf_category']=='Immune modulation') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory']='unknown_IM_subcat'
    search_res.loc[(search_res['tvf_category']=='Exoenzyme') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory'] = search_res.loc[(search_res['tvf_category']=='Exoenzyme') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory']='unknown_Exoenzyme_subcat'
    search_res.loc[(search_res['tvf_category']=='Invasion') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory'] = search_res.loc[(search_res['tvf_category']=='Invasion') & (search_res['tvf_subcategory']=='non_vf'),'tvf_subcategory']='unknown_Invasion_subcat'

    tcat_mask = search_res['tvf_category'] == 'Effector delivery system'
    search_res.loc[tcat_mask,'tvf_category'] = search_res.loc[tcat_mask,'tvf_subcategory']

    tcat_mask = search_res['tvf_category'] == 'Exoenzyme'
    search_res.loc[tcat_mask,'tvf_category'] = 'Exotoxin'

    # labels in df need to match labels in label binarizer from glm classification
    # search_res['tvf_category'] = search_res['tvf_category'] + ' '
    # search_res['tvf_category'] = search_res['tvf_category'].str.replace('non_vf ','non_vf') # non_vfs don't have an additional space in lb
    return search_res


def vfcat_score(allvall_sub_sub, score_metric):
    """Get the top hit for each category based on score_metric"""
    allvall_sub_sub = allvall_sub_sub.reset_index(drop=True)
    tophits = allvall_sub_sub.drop_duplicates(subset=['tvf_category'], keep="first")
    vfcat_scores = dict(zip(tophits['tvf_category'], tophits[score_metric]))
    return vfcat_scores

def alltopNhits_probs_groupby(group, score_metric, lb):
    """Process a single query group - optimized for groupby"""
    # Sort within the group
    group_sorted = group.sort_values(by=score_metric, ascending=False) # can probably just sort before groupby
    
    # Get category scores
    vfcat_probs = vfcat_score(group_sorted, score_metric)
    
    # Build predictions array
    preds = []
    for vfcat in vfcat_probs:
        label = lb.transform([vfcat]).astype(np.float64)
        label[label > 0] = vfcat_probs[vfcat]
        preds.append(label[0])
    
    preds = np.array(preds)
    preds = np.sum(preds, axis=0)
    
    if np.sum(preds) == 0:
        query_rep = group['query'].iloc[0]
        print(f"--Getting no structural similarity to db for {query_rep}--")
    else:
        preds = preds / np.sum(preds)  # normalize
    
    return preds

def format_strucpreds(pred_raw,lb):
    # scale structure predictions and to a dataframe, then map back query protein labels

    struct_preds = pd.DataFrame(pred_raw)
    struct_preds_expanded = pd.DataFrame(struct_preds[struct_preds.columns[0]].tolist(),index=struct_preds.index)

    # scale structure-based predictions so the LR ensemble model has an easier time training
    scaler = StandardScaler()
    scaler.fit(struct_preds_expanded.values)
    struct_predictions_scaled = scaler.transform(struct_preds_expanded.values)
    struct_predictions_scaled = pd.concat([pd.DataFrame(struct_predictions_scaled),struct_preds.iloc[:,1]],axis=1)
    # print("struct predictions scaled post concat (line 185)", struct_predictions_scaled.head(),flush=True)
    
    # get names of of columns in predictions
    col_names = [cat for cat in lb.classes_]
    col_names.append('query')
    struct_predictions_scaled.columns = col_names
    # print("struct predictions scaled post concat with col names (line 191)", struct_predictions_scaled.head(),flush=True)
    return struct_predictions_scaled

# call spacedust

