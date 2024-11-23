import pandas as pd
import pickle as pkl
import importlib.resources

def load_vf_functional_mappers():
    # Get the TSV file as a stream
    data_dir =  importlib.resources.open_text("meetneighbors.predictvfs.data.vf_functional_mappers")

    dfs = {file.name:pd.read_csv(file.open("r",encoding="utf-8"),sep="\t") for file in data_dir.iterdir() if file.name.endswith('.tsv')}

    dfs['VFID_mapping_specified.tsv'] = dfs['VFID_mapping_specified.tsv'].apply(lambda x: x.str.strip() if x.dtype == "object" else x) # format mapping file
    dfs['vfquery_to_id_tocat.tsv'] = dict(zip(dfs['vfquery_to_id_tocat.tsv']['query'],dfs['vfquery_to_id_tocat.tsv']['VFID'])) # send tsv to dictionary, only need 2 cols of info
    return dfs

def load_pickle():
    # Open the pickled file as a binary stream
    data_dir = importlib.resources.open_binary("meetneighbors.predictvfs.data.pkl_objs")
    # Deserialize the pickled data
    data = {file.name:pkl.load(file.open('rb')) for file in data_dir if file.name.endswith(".obj")}
    return data

def load_nn_clf_model():
    # return path of model for pytorch to work with
    return importlib.resources.path("src.meet-neighbors.subpackage.predictvfs.data","nn_clf_pytorch_11212024.bin")