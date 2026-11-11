import pandas as pd
import numpy as np
import glob
import os
import pickle as pkl
from tqdm import tqdm


def legacy_center_targets(rows, hashes, mapping):
    """Historical hash/rep matching: retain the first flattened occurrence.

    Three-field mappings do not identify repeated occurrences. Keep the old
    compute_umap selection for these artifacts only.
    """
    from meetneighbors.neighborhood_order import query_labels
    first_occurrence = {}
    for i, row in enumerate(rows):
        key = hashes[row.neighborhood_name]
        for position, protein in enumerate(row.proteins):
            first_occurrence.setdefault((key, protein[1:]), (i, position))
    targets = []
    first_rep, seen = {}, set()
    for key, entries in mapping.items():
        for query, rep, name in entries:
            first_rep.setdefault(name, rep)
            occurrence = first_occurrence.get((key, rep))
            if first_rep[name] != rep or occurrence is None:
                continue
            for label in query_labels([query]):
                if (label, name) not in seen:
                    targets.append((*occurrence, label, rep, name))
                    seen.add((label, name))
    return targets


def unpack_embeddings(glm_out_dir,glm_in_dir,mmseqs_clust,logger=None, **kwargs): # lowkey don't need mmseqs_clust but it's here just to make sure things are working
    from meetneighbors.embedding_pipeline import saved_batch_paths
    batch_files = saved_batch_paths(glm_out_dir)
    if batch_files:
        return {str(i): pd.read_csv(path, sep='\t') for i, path in enumerate(batch_files)}
    mem_optimize = kwargs.get('mem_optimize',None)
    if mem_optimize:
        from meetneighbors.embedding_pipeline import hash_format
        columns = ['nn_hashes', 'query', 'rep', 'neighborhood_name']
        if hash_format(mem_optimize) == 'indexed':
            columns.append('nn_idx')
        nn_hash_df = pd.DataFrame([(k, *vals) for k, lst in mem_optimize.items() for vals in lst], columns=columns)
        mem_optimize = True # here cus im tired, instead of making new func input for nn_hash 
        # query_rep_d = nn_hash_df.groupby('rep')['query'].apply(set).to_dict() # this is a dict of query to set of reps that are in the gLM batch for that query, this is used to subset the nn_hash_df for only the reps that are in the gLM batch for each query

    glm_folders_o = [folder for folder in glob.glob(glm_out_dir+"/*") if os.path.isdir(folder)]
    glm_vf_fldrs_o = {fold.split('/')[-1]:fold for fold in glm_folders_o}
    
    glm_folders_i = [folder for folder in glob.glob(glm_in_dir+"/*.tsv")]
    glm_vf_fldrs_i = {fold.split('/')[-1].split('.ts')[0]:fold for fold in glm_folders_i}

    glm_res_d_vals_predf = {}
    # below column names are used for virulent mode, 
    input_tsv_cols = ['neighborhood_name','rep','vf_center_index']

    def get_hashedout_embeds(nn_hash_df,mmseqs_clust,glm_batch,input_tsv):
        if 'nn_idx' in nn_hash_df:
            from meetneighbors.glm_input_frm_neighbors import Neighborhood
            from meetneighbors.embedding_pipeline import center_targets, METADATA
            rows, offsets, offset = [], [], 0
            for record in input_tsv.itertuples():
                proteins = tuple(record.rep.split(';'))
                rows.append(Neighborhood(glm_vf_fldrs_i[vf], record.neighborhood_name, (),
                                         proteins, int(record.vf_center_index),
                                         proteins[int(record.vf_center_index)][1:]))
                offsets.append(offset)
                offset += len(proteins)
            if offset != len(glm_batch):
                raise ValueError('Legacy protein embedding count does not match GLM inputs')
            hashes = mmseqs_clust.loc[mmseqs_clust.neighborhood_name.isin(input_tsv.neighborhood_name), 'nn_hashes']
            mapping = {key: group[['query', 'rep', 'neighborhood_name', 'nn_idx']].values.tolist()
                       for key, group in nn_hash_df[nn_hash_df.nn_hashes.isin(hashes)].groupby('nn_hashes')}
            targets = center_targets(rows, mmseqs_clust, mapping)
            values = []
            for i, position, metadata in targets:
                rep, vector = glm_batch[offsets[i] + position]
                if rep != metadata[3]:
                    raise ValueError('Legacy embedding protein does not match indexed center')
                values.append(vector)
            return pd.concat([pd.DataFrame([m for _, _, m in targets], columns=METADATA),
                              pd.DataFrame(np.asarray(values))], axis=1)
        input_tsv['rep'] = input_tsv['rep'].str.split(';')
        input_tsv = input_tsv.explode('rep').reset_index(drop=True).reset_index(names='rep_index') # now rep_index numbers should line up with what's in glm_batch
        input_tsv['rep'] = input_tsv['rep'].str[1:]

        mmseqs_clust = mmseqs_clust.loc[mmseqs_clust['prot_gffname'] == (mmseqs_clust['VF_center'] +'!!!'+ mmseqs_clust['gff'])] # subset mmseqs_clust for rows with VF centers

        input_tsv = pd.merge(input_tsv,mmseqs_clust[['neighborhood_name','nn_hashes']],on='neighborhood_name',how='left')
        del mmseqs_clust
        nn_hash_df = nn_hash_df[nn_hash_df['nn_hashes'].isin(set(input_tsv['nn_hashes']))] # need to subset since not all hashes are in one chunk of glm_inputs
        nn_hash_df_m = pd.merge(nn_hash_df,input_tsv[['rep_index','nn_hashes','rep']],on='nn_hashes',how='left')
        del input_tsv
        nn_hash_df_m.drop_duplicates(subset=['neighborhood_name','rep_index','rep_y'],inplace=True)
        nn_hash_df_m = nn_hash_df_m.loc[nn_hash_df_m['rep_x']==nn_hash_df_m['rep_y']] # this works because rep_x are only reps that belonged to a query prot
        nn_hash_df_m.reset_index(drop=True,inplace=True) # need to reset_index b/c pd.concat will create NAs if not
        # if memory fails again maybe try converting the necessary input_tsv cols into a numpy arrays
        rep_inds = nn_hash_df_m['rep_index'].values
        glm_batch = np.array(glm_batch, dtype=object)
        glm_batch = pd.DataFrame(np.vstack(glm_batch[rep_inds][:,1])) # got to vstack b/c glm_batch[rep_inds] makes an array of arrays and not just a normal 2D array
        glm_embeds_df_piece = pd.concat([nn_hash_df_m[['neighborhood_name','rep_y']],glm_batch],axis=1) 
        glm_embeds_df_piece = pd.merge(glm_embeds_df_piece,nn_hash_df.rename(columns={'rep':'rep_y'})[['query','rep_y','neighborhood_name']],on=['neighborhood_name','rep_y'],how='left') # merging with originial (on purpose) nn_hash_df pre merge to get query labels. See comments below
        glm_embeds_df_piece = glm_embeds_df_piece.explode('query').reset_index(drop=True) # this is necessary because some queries have multiple reps in the gLM batch, so we need to explode the query column to match the number of rows in the glm_batch
        glm_embeds_df_piece.drop_duplicates(subset=['query','neighborhood_name'],inplace=True)

        # keep the below comments for future ref
        # current issue is that sometimes the same rep can be a query in one nn, and not in another
        # this is b/c query labels are mapped by lc and not rep
        # this issue then causes the over representation of reps at unnecessary instances (when their not a query) in the final res
        # solution: i think I need to keep the rep of the vf_center in nn_hash_d
        # the input_tsv links that rep to the nn to the hash, i need to merge with nn_hash_df by hash
        # for each nn in the recent merge grab the rep of the vf_center or where nn_hash_df rep = input_tsv rep

        return glm_embeds_df_piece
    
    query_reps = mmseqs_clust.dropna(subset='query').groupby('neighborhood_name')['rep'].apply(set).to_dict() # some queries may not be the VF center of the neighborhoodname key, but that's ok for this usecase
    
    for vf in tqdm(glm_vf_fldrs_o):
        input_tsv = pd.read_csv(glm_vf_fldrs_i[vf],sep="\t",names=input_tsv_cols) #read in gLM input
        batch_size = [len(row.rep.split(';')) for row in input_tsv.itertuples()] # get the whole size of each neighborhood for later indexing for VF center
        # create additional rows of nns with a hash?
        glm_index = [int(input_tsv.vf_center_index.iloc[i]) + sum(batch_size[:i]) for i in range(len(batch_size))] # grabbing index of protein in gLM batch output obj
        input_tsv['glm_indicies'] = glm_index
        glm_batch = pkl.load(open(glm_vf_fldrs_o[vf]+'/results/results/batch.pkl.glm.embs.pkl','rb'))
        if mem_optimize:
            logger.debug("Grabbing embedddings for hashed out neighborhoods")
            glm_res_d_vals_predf[vf] = get_hashedout_embeds(nn_hash_df,mmseqs_clust,glm_batch,input_tsv)
        else:
            glm_batch = np.array(glm_batch, dtype=object)[input_tsv['glm_indicies'].values] #subset gLM batch for indicies of VF centers
            assert len(glm_batch) == len(input_tsv),"gLM batch size subset and input tsv size do not match"
            for i,embed in enumerate(glm_batch): # there should be an embedding for each row in input_tsv or for each VF center
                glm_res_d_vals_predf[vf+"!!!"+str(i)] = np.append(embed[1],input_tsv.iloc[i].loc[['neighborhood_name']].values) # used i to index input_tsv instead of 0 b/c neighborhood names are different for each row
                assert embed[0] in query_reps[input_tsv['neighborhood_name'].iloc[i]], f"Embedding {embed[0]} in gLM batch {vf} is not a rep of a target VF." # make sure 

    return glm_res_d_vals_predf

def get_glm_embeddf(glm_res_d_vals_predf,**kwargs): # format dictionary to an embedding df for uMAP plotting
    mem_optimize = kwargs.get('mem_optimize',None)
    if mem_optimize or all(isinstance(v, pd.DataFrame) for v in glm_res_d_vals_predf.values()):
        embedding_df = pd.concat([glm_res_d_vals_predf[k] for k in glm_res_d_vals_predf],axis=0)
        # assert len(embedding_df[embedding_df['query'].isna()]) == 0, "Some queries are NA, pls fix" # need to unhash after I remove nns with no query from merge
    else:
        embedding_df = pd.DataFrame.from_dict(glm_res_d_vals_predf,orient='index')
        embedding_df.reset_index(names="query",inplace=True) # names parameter here only works on newer versions of pandas
        embedding_df["query"] = embedding_df["query"].str.split('!!!').str[0]
        cols_names = ['neighborhood_name']
        col_names_map = {col:cols_names[i] for i, col in enumerate(embedding_df.columns[-len(cols_names):])}
        embedding_df.rename(columns=col_names_map, inplace=True)
    return embedding_df

def get_umapdf(embedding_df):
    import umap # i know it's weird but this package is a pain to install

    cols_names = ['neighborhood_name','vf_name','vfid','vf_category','species','genus']
    from meetneighbors.embedding_pipeline import embedding_columns
    umapper = umap.UMAP().fit(embedding_df[embedding_columns(embedding_df)].values)
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
