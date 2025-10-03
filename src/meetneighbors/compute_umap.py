import pandas as pd
import numpy as np
import glob
import os
import pickle as pkl
from tqdm import tqdm


def unpack_embeddings(glm_out_dir,glm_in_dir,mmseqs_clust, **kwargs): # lowkey don't need mmseqs_clust but it's here just to make sure things are working
    mem_optimize = kwargs.get('mem_optimize',None)
    if mem_optimize:
        nn_hash_df = pd.DataFrame([(k, *vals) for k, lst in mem_optimize.items() for vals in lst],columns=['nn_hashes','query','neighborhood_name'])
        mem_optimize = True # here cus im tired, instead of making new func input for nn_hash 

    glm_folders_o = [folder for folder in glob.glob(glm_out_dir+"/*") if os.path.isdir(folder)]
    glm_vf_fldrs_o = {fold.split('/')[-1]:fold for fold in glm_folders_o}
    
    glm_folders_i = [folder for folder in glob.glob(glm_in_dir+"/*.tsv")]
    glm_vf_fldrs_i = {fold.split('/')[-1].split('.ts')[0]:fold for fold in glm_folders_i}

    glm_res_d_vals_predf = {}
    # below column names are used for virulent mode, 
    input_tsv_cols = ['neighborhood_name','neighborhood','vf_center_index']

    # def get_hashed_embeddings(hash,nn_hash_d,glm_batch,input_tsv): #for hashed nns

    def get_hashedout_embeds(nn_hash_df,mmseqs_clust,glm_batch,input_tsv,glm_res_d_vals_predf):
        input_tsv['neighborhood'] = input_tsv['neighborhood'].str.split(';')
        input_tsv = input_tsv.explode('neighborhood').reset_index(drop=True).reset_index(names='rep_index') # now rep_index numbers should line up with what's in glm_batch
        input_tsv['neighborhood'] = input_tsv['neighborhood'].str[1:]

        mmseqs_clust = mmseqs_clust.loc[mmseqs_clust['prot_gffname'] == (mmseqs_clust['VF_center'] +'!!!'+ mmseqs_clust['gff'])] # subset mmseqs_clust for rows with VF centers

        input_tsv = input_tsv[input_tsv['neighborhood'].isin(set(mmseqs_clust['rep']))] # now its only VF centers in glm_inputs
        input_tsv = pd.merge(input_tsv,mmseqs_clust[['neighborhood_name','nn_hashes']],on='neighborhood_name',how='left') 
        nn_hash_df = nn_hash_df[nn_hash_df['nn_hashes'].isin(set(input_tsv['nn_hashes']))] # need to subset since not all hashes are in one chunk of glm_inputs
        nn_hash_df = pd.merge(nn_hash_df,input_tsv[['neighborhood','rep_index','nn_hashes']],on='nn_hashes',how='left')

        for i,row in enumerate(nn_hash_df.itertuples()):
            try:
                glm_res_d_vals_predf[row.query + '!!!' + str(i)] = np.append(glm_batch[row.rep_index][1], row.neighborhood_name)
            except Exception as e:
                print(e)
                print(row)
                input_tsv.to_csv('glm_inputdf.tsv',sep="\t",index=False)
                nn_hash_df.to_csv('nn_hashdf.tsv',sep="\t",index=False)
                mmseqs_clust.to_csv('mmseqs_clust_sub.tsv',sep="\t",index=False)
                raise Exception
        return glm_res_d_vals_predf
    # maybe make a df with columns: rep,nn from input_tsv, index: 0-n_reps, index_matches those in glm_outputs
    # map df indices to reps in mmseqs_clust_sub
    # filter mmseqs_clust_sub for hashes in input_tsv, but remove nns from mmseqs_clust_sub that are in input_tsv
    # now subset glm_batch for indices in mmseqs_clust_sub
    
    query_reps = mmseqs_clust.dropna(subset='query').groupby('neighborhood_name')['rep'].apply(set).to_dict() # some queries may not be the VF center of the neighborhoodname key, but that's ok for this usecase
    
    for vf in tqdm(glm_vf_fldrs_o):
        input_tsv = pd.read_csv(glm_vf_fldrs_i[vf],sep="\t",names=input_tsv_cols) #read in gLM input
        batch_size = [len(row.neighborhood.split(';')) for row in input_tsv.itertuples()] # get the whole size of each neighborhood for later indexing for VF center
        # create additional rows of nns with a hash?
        glm_index = [int(input_tsv.vf_center_index.iloc[i]) + sum(batch_size[:i]) for i in range(len(batch_size))] # grabbing index of protein in gLM batch output obj
        input_tsv['glm_indicies'] = glm_index
        glm_batch = pkl.load(open(glm_vf_fldrs_o[vf]+'/results/results/batch.pkl.glm.embs.pkl','rb'))
        if mem_optimize:
            glm_res_d_vals_predf = get_hashedout_embeds(nn_hash_df,mmseqs_clust,glm_batch,input_tsv,glm_res_d_vals_predf)
        else:
            glm_batch = np.array(glm_batch, dtype=object)[input_tsv['glm_indicies'].values] #subset gLM batch for indicies of VF centers
            assert len(glm_batch) == len(input_tsv),"gLM batch size subset and input tsv size do not match"
            for i,embed in enumerate(glm_batch): # there should be an embedding for each row in input_tsv or for each VF center
                glm_res_d_vals_predf[vf+"!!!"+str(i)] = np.append(embed[1],input_tsv.iloc[i].loc[['neighborhood_name']].values) # used i to index input_tsv instead of 0 b/c neighborhood names are different for each row
                assert embed[0] in query_reps[input_tsv['neighborhood_name'].iloc[i]], f"Embedding {embed[0]} in gLM batch {vf} is not a rep of a target VF." # make sure 

    return glm_res_d_vals_predf

def get_glm_embeddf(glm_res_d_vals_predf): # format dictionary to an embedding df for uMAP plotting
    
    embedding_df = pd.DataFrame.from_dict(glm_res_d_vals_predf,orient='index')
    embedding_df.reset_index(names="query",inplace=True) # names parameter here only works on newer versions of pandas
    embedding_df["query"] = embedding_df["query"].str.split('!!!').str[0]
    cols_names = ['neighborhood_name']
    col_names_map = {col:cols_names[i] for i, col in enumerate(embedding_df.columns[-len(cols_names):])}
    embedding_df.rename(columns=col_names_map, inplace=True)
    # embedding_df.columns.values[-len(cols_names):] = cols_names # I think this really messed up the col names so that I couldn't call it in the future
    return embedding_df

def get_umapdf(embedding_df):
    import umap # i know it's weird but this package is a pain to install

    cols_names = ['neighborhood_name','vf_name','vfid','vf_category','species','genus']
    umapper = umap.UMAP().fit(embedding_df.iloc[:,1:-len(cols_names)].values)
    return umapper

def plt_baby(umapper,embedding_df_merge,**kwargs):
    outdir = kwargs.get("outdir")
    plt_name = kwargs.get("plt_name")
    legend = kwargs.get("legend")
    width = kwargs.get("width")
    label = kwargs.get("label")
    legend = kwargs.get("legend")
    import umap
    import umap.plot # may need to conda install the requirements

    if not os.path.isfile(f"{outdir}{plt_name}.tsv"):
        embedding_df_merge.to_csv(f"{outdir}{plt_name}.tsv",sep="\t",index=False,mode="x")

    p = umap.plot.points(umapper, labels=embedding_df_merge[label],show_legend=legend,width=width)
    p_obj = p.get_figure()
    assert not os.path.isfile(f"{outdir}{plt_name}.png"), "File with this name already exists"
    p_obj.savefig(f"{outdir}{plt_name}.png")
    return 
