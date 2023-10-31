import pandas as pd
import _pickle as cPickle
import time
from pathlib import Path
from Bio import SeqIO
import dask 
import dask.bag as db

from process_gffs import get_neighborhoodIDs_wGFF
from process_gffs import get_protseq_frmFasta

def get_allgffs(path_to_gffs,pickle_path):
    # I should implement this w/ dask instead of multiprocessing
    start_time = time.time()
    gffs = glob.glob(path_to_gffs)
    gffs = db.from_sequence(gffs[:200])
    all_gffs = db.map(gff2pandas,gffs)
    all_gffs.persist()
    pickle_file = Path(pickle_path)
    if not pickle_file.is_file(): #check if pickle file exists so it doesn't get written over
        all_gffs.compute()
        fileObj = open(pickle_path, 'wb')
        pickle.dump(all_gffs, fileObj)
        fileObj.close()
    else: print('pickle file with same name already exists')
    print(f"--- %s seconds --- for {len(gffs)} gffs" % (time.time() - start_time))
    return all_gffs

def run_neighborhoods(gff_df,vf_fasta_withIDs,pickle_neighborhood): 
    start_time = time.time()
    #get protein ids from vfdb fasta
    vf_fasta = SeqIO.parse(vf_fasta_withIDs,'fasta')
    ids =  list(map(lambda x: x.id.split('gb|')[1].split(')')[0],vf_fasta))
    print('!!! VF IDs identified!!!')
    if not isinstance(gff_df, pd.DataFrame):
        assert Path(gff_df).is_file()
        file = open(gff_df, 'rb')
        all_gffs = cPickle.load(file)
        file.close()
        print('!!! GFF df loaded from pickle!!!')
        all_gffs = db.from_sequence(all_gffs)
    neighborhood_db = db.map(get_neighborhoodIDs_wGFF,ids,all_gffs,10000)
    del all_gffs #remove this variable cus it prolly takes up an f ton of memory
    neighborhood_db = neighborhood_db.filter(lambda x: x is not None).persist()
    neighborhood_db = neighborhood_db.filter(lambda x: len(x)>0).persist()
    neighborhood_db = neighborhood_db.flatten().persist()
    print(f'!!! {len(list(neighborhood_db))} Neighborhoods gathered!!!')

    if not Path(pickle_neighborhood).is_file(): #might be a good idea to make this if statement a decorator or a separate function later on
        neighborhood_db_to_save = neighborhood_db.compute()
        fileObj = open(pickle_neighborhood, 'wb')
        cPickle.dump(neighborhood_db_to_save, fileObj)
        fileObj.close()
    print(f"--- %s seconds --- for run_neighborhoods()" % (time.time() - start_time))
    return neighborhood_db


def run_fasta_from_neighborhood(dir_for_fasta,neighborhood,fltr_neighborhoods):
    #modify this func to output fastas in a separate directory
    start_time = time.time()
    og_neighborhood_size = len(list(neighborhood))
    if fltr_neighborhoods: #remove neighborhoods with only one protein
        neighborhood.filter(lambda x: len(x) <= 1).persist()
        print(f'{og_neighborhood_size - len(list(neighborhood))} Neighborhoods filtered because there was 1 protein or less')
    run_fasta = db.map(get_protseq_frmFasta,dir_for_fasta,neighborhood,to_fasta=False)
    run_fasta.compute()
    print(f"--- %s seconds --- for creating fastas" % (time.time() - start_time))
    return

def run_all():
    if __name__ == "__main__":
        all_gffs = get_allgffs(path_to_gffs='/gpfs/data/pirontilab/Students/Madu/bigdreams_dl/testrun/ab_fastas_gffs/*.gff',pickle_path='ab_gffs_df.obj')
        neighborhood_db = run_neighborhoods(gff_df=all_gffs,vf_fasta_withIDs='/gpfs/data/pirontilab/Students/Madu/bigdreams_dl/vfdb_ab_sub.fasta',pickle_file_name='neighborhoods_gathrd_allab.obj')
        run_fasta_from_neighborhood(dir_for_fasta='/gpfs/data/pirontilab/Students/Madu/bigdreams_dl/testrun/ab_fastas_gffs/',neighborhood=neighborhood_db,fltr_neighborhoods=True)

run_all()
