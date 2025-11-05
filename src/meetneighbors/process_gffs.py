import pandas as pd
import numpy as np
import gffpandas.gffpandas as gffpd
import time
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import traceback
import os

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
    # this function loads in a gff into a pandas dataframe, then subset for only necessary columns
    
    try: 
        df = gffpd.read_gff3(gff)
        df = df.attributes_to_columns().loc[df.attributes_to_columns()['type']=='CDS']
        #df['RefSeq'] = df.inference.str.split('RefSeq:').str[-1]
        df['gff_name'] = gff.split('/')[-1].split('_genomic.gff')[0]
        cols_to_drop = ['bound_moiety','score','source','regulatory_class','start_range','strain', 'transl_table','gene_biotype', 'gene_synonym',
            'go_component', 'go_function', 'go_process','isolation-source','mol_type', 'nat-host',
                        'collection-date', 'country', 'end_range','serovar',
            'exception','anticodon','phase','Ontology_term','attributes','gene','Dbxref','Name','ID','inference','collected-by','isolate', 'lat-lon']
        df.drop(columns=df.columns.intersection(cols_to_drop),inplace=True) #remove col if it exists in df
    except AttributeError:
        print(f'gff2pandas caught an error for this {gff} gff')
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
            matching_rows = group_data.copy() #this should prevent the pandas copy warning
            matching_rows = group_data[
                (group_data['start'] >= row.start - window) &
                (group_data['end'] <= row.end + window)]

            #result_ids = matching_rows['RefSeq'].tolist()
            matching_rows['VF_center'] = row.RefSeq
            neighborhoods.append(matching_rows)
    except KeyError:
        return None
    return neighborhoods

    
def get_protseq_frmFasta(logger, args, dir_for_fastas, neighborhood):
    """Optimized version with pre-computed lookups and efficient filtering."""
    rec = None
    try:
        # Pre-compute values used multiple times
        first_row = neighborhood.iloc[0]
        last_row = neighborhood.iloc[-1]
        
        # Define neighborhood naming schema
        neighborhood_name = f'{first_row.VF_center}!!!{first_row.gff_name}!!!{first_row.seq_id}!!!{first_row.start}-{last_row.end}'
        
        # Convert protein_id to set for O(1) lookups instead of O(n)
        protein_ids = set(neighborhood.protein_id)
        
        # Pre-build lookup dictionary for start positions and strands (avoid repeated DataFrame queries)
        protein_lookup = {
            row.protein_id+'!!!'+str(i): (row.start, row.strand)
            for i,row in enumerate(neighborhood.itertuples())
        }
        
        # Determine fasta directory
        if isinstance(args.genome_tsv, pd.DataFrame):
            fasta_dir = args.genome_tsv[args.genome_tsv['genome'] == first_row.gff_name]['protein'].iloc[0]
        else:
            if not dir_for_fastas.endswith('/'):
                dir_for_fastas += '/'
            
            # Try different filename patterns
            for suffix in ['_protein.faa', '.faa', '.fasta']:
                fasta_dir = dir_for_fastas + first_row.gff_name + suffix
                if os.path.isfile(fasta_dir):
                    break
            else:
                raise AssertionError(f"Protein file for {dir_for_fastas + first_row.gff_name} could not be found.")
        
        # Parse and filter FASTA records
        rec = {}
        for record in SeqIO.parse(fasta_dir, "fasta"):
            # Extract ID after last '|'
            rec_id = record.id.split('|')[-1]
            
            if rec_id in protein_ids:
                # Create new record with modified ID
                # new_rec = SeqRecord(record.seq, id=rec_id, description=record.description)
                record.id = rec_id
                rec[record.id] = [record.seq,record.description.split(f'{record.id} ')[1]]
                
                # Early exit if we've found all proteins
                if len(rec) == len(protein_ids):
                    break
        assert len(rec) == len(protein_ids), f"Protein ID from nn {neighborhood_name} is missing from {fasta_dir}"

        # # Check protein count constraints
        # if len(rec) < args.min_prots or len(rec) > args.max_prots:
        #     return []
        
        # Build final records with enhanced IDs
        result = [SeqRecord(id = f"{p.split('!!!')[0]}!!!{neighborhood_name}!!!{protein_lookup[p][0]}!!!{protein_lookup[p][1]}",
                            description=rec[p.split('!!!')[0]][1], seq=rec[p.split('!!!')[0]][0]).format("fasta")[:-1] 
                            for p in protein_lookup]
        
        return result
        
    except AttributeError:
        traceback.print_exc()
        logger.error('Item did not contain neighborhood')
        return
