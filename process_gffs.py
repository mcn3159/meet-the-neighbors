import pandas
import numpy as np
import gffpandas.gffpandas as gffpd
import time
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pathlib import Path

def first_decorator(func):
    def wrapper(*args, **kwarg):
        try:
            func(*args, **kwarg)
            return func(*args, **kwarg)
        except AttributeError:
            print(*args)
    return wrapper

#@first_decorator
def gff2pandas(gff):
    try: 
        df = gffpd.read_gff3(gff)
        df = df.attributes_to_columns().loc[df.attributes_to_columns()['type']=='CDS']
        df['RefSeq'] = df.inference.str.split('RefSeq:').str[-1]
        df['gff_name'] = gff.split('/')[-1].split('_genomic.gff')[0]
        cols_to_drop = ['bound_moiety','score','source','regulatory_class','start_range','strain', 'transl_table','gene_biotype', 'gene_synonym',
            'go_component', 'go_function', 'go_process','isolation-source','mol_type', 'nat-host',
                        'collection-date', 'country', 'end_range','serovar',
            'exception','anticodon','phase','Ontology_term','attributes','gene','Dbxref','Name','ID','inference','collected-by','isolate', 'lat-lon']
        df.drop(columns=df.columns.intersection(cols_to_drop),inplace=True) #remove col if it exists in df
    except AttributeError:
        print(gff)
    return df

def get_neighborhoodIDs_wGFF(specific_ids,df,window):
    try:
        # currently this function assumes that there is no duplicate VFs on the same contig strand and gff file
        specific_df = df[df['RefSeq'].isin(specific_ids)]
        # Group the DataFrame by the 'D' column (specified IDs)
        grouped = df.groupby(['gff_name', 'strand', 'seq_id'])
        neighborhoods = []
        for row in specific_df.itertuples():
            group_data = grouped.get_group((row.gff_name,row.strand,row.seq_id))
            matching_rows = group_data[
                (group_data['start'] >= row.start - window) &
                (group_data['end'] <= row.end + window)]

            #result_ids = matching_rows['RefSeq'].tolist()
            matching_rows['VF_center'] = row.RefSeq
            neighborhoods.append(matching_rows)
    except KeyError:
        return None
    return neighborhoods

def getCDSneighbors(IDs,df):
    start_time = time.time()
    neighborhoods_l = []
    for i in IDs:
        df_sub = df.loc[df['RefSeq'] == i]
        start,end,strand,contig,gcf = df_sub.iloc[0].start,df_sub.iloc[0].end,df_sub.iloc[0].strand,df_sub.iloc[0].seq_id,df_sub.iloc[0].gff_name
        neighborhood_df = df[(df['start'] > start-10000) & (df['end'] < end+10000) &
                             (df['strand'] == strand) & (df['seq_id'] == contig) & (df['gff_name'] == gcf)]
        neighborhoods_l.append(neighborhood_df)
    print(f"--- %s seconds ---" % (time.time() - start_time))
    return neighborhoods_l

def get_protseq_frmFasta(dir_for_fastas,neighborhood,to_fasta):
    try:
        row_with_vf = neighborhood[neighborhood['RefSeq'] == neighborhood['VF_center'].iloc[0]]
        fasta_dir = dir_for_fastas+neighborhood.iloc[0].gff_name+'_protein.faa'
        fasta = SeqIO.parse(fasta_dir,'fasta')
        rec = filter(lambda x: x.id in list(neighborhood.protein_id),fasta)
        rec = map(lambda x: SeqRecord(x.seq,id=
                                      x.id+f'||||{neighborhood.iloc[0].VF_center}||||{neighborhood.iloc[0].gff_name}||||{neighborhood.iloc[0].seq_id}',description=x.description.split(f'{x.id} ')[1]),rec)
        if to_fasta: #make sure to_fasta is either False, or the name of the file when running
            neighborhood_fasta_name = row_with_vf.iloc[0].VF_center+'||||'+row_with_vf.iloc[0].gff_name+\
                                      '||||'+row_with_vf.iloc[0].seq_id+'||||'+str(row_with_vf.iloc[0].start)+'||||'+str(row_with_vf.iloc[0].end)+'.faa'
            if Path(neighborhood_fasta_name).is_file():
                print('Neighborhood fasta with same name already exists')
            else:
                with open(neighborhood_fasta_name, 'w') as handle:
                    SeqIO.write(rec, handle, "fasta")
    except AttributeError:
        print('Item did not contain neighborhood')
        return
    return

