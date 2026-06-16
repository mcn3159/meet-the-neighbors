import pandas as pd
import pickle as pkl
import importlib.resources
import os
from pathlib import Path

GLM_MODEL_ENV_VAR = "MEETNEIGHBORS_GLM_BIN"
GLM_MODEL_FILENAME = "glm.bin"
GLM_MODEL_REPO_RELATIVE_PATH = Path("src/meetneighbors/predictvfs/glm/model") / GLM_MODEL_FILENAME

def load_vf_functional_mappers():
    # Get the TSV file as a stream
    data_dir =  importlib.resources.files("meetneighbors.predictvfs.data.vf_functional_mappers")

    dfs = {file.name:pd.read_csv(file.open("r",encoding="utf-8"),sep="\t") for file in data_dir.iterdir() if file.name.endswith('.tsv')}

    dfs['VFID_mapping_specified.tsv'] = dfs['VFID_mapping_specified.tsv'].apply(lambda x: x.str.strip() if x.dtype == "object" else x) # format mapping file
    dfs['vfquery_to_id_tocat.tsv'] = dict(zip(dfs['vfquery_to_id_tocat.tsv']['query'],dfs['vfquery_to_id_tocat.tsv']['VFID'])) # send tsv to dictionary, only need 2 cols of info
    return dfs

def load_pickle():
    # Open the pickled file as a binary stream
    data_dir = importlib.resources.files("meetneighbors.predictvfs.data.pkl_objs")
    # Deserialize the pickled data
    data = {file.name:pkl.load(file.open('rb')) for file in data_dir.iterdir() if file.name.endswith(".obj") or file.name.endswith(".pkl")}
    return data

def load_clf_models():
    # return path of model for pytorch to work with
    models_dict = {"nn_clf":importlib.resources.path("meetneighbors.predictvfs.models","glm_classifier_nnsclust_07threshavg_5152026.bin"),
                   "int_clf":importlib.resources.path("meetneighbors.predictvfs.models","meta-LR_12222025.obj")}
    return models_dict

def get_glm_model_path():
    """Return the local path to glm.bin or raise a setup-focused error."""
    env_path = os.environ.get(GLM_MODEL_ENV_VAR)
    if env_path:
        model_path = Path(env_path).expanduser()
        if model_path.is_file():
            return model_path
        raise FileNotFoundError(
            f"{GLM_MODEL_ENV_VAR} is set to '{env_path}', but that file does not exist. "
            f"Set {GLM_MODEL_ENV_VAR} to a valid glm.bin path, or copy glm.bin to "
            f"{GLM_MODEL_REPO_RELATIVE_PATH} in the cloned repo."
        )

    for parent in [Path.cwd(), *Path.cwd().parents]:
        model_path = parent / GLM_MODEL_REPO_RELATIVE_PATH
        if model_path.is_file():
            return model_path

    installed_adjacent_path = Path(__file__).resolve().parent / "glm" / "model" / GLM_MODEL_FILENAME
    if installed_adjacent_path.is_file():
        return installed_adjacent_path

    raise FileNotFoundError(
        "Missing gLM model weights: glm.bin. This file is intentionally not "
        "installed with the package because it is large. To use "
        "`meetneighbors predictvf`, copy/download glm.bin to "
        f"{GLM_MODEL_REPO_RELATIVE_PATH} in the cloned repo and run commands "
        f"from that repo, or set {GLM_MODEL_ENV_VAR}=/path/to/glm.bin. "
        "If you use the repo-local path or environment variable, `pip install .` "
        "does not need to be rerun after copying the file."
    )
    
