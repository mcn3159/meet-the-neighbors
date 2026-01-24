# Copyright (c) President and Fellows of Harvard College 2023.

# This source code is licensed under the Academic and Non-Commercial Research Use Software License 
# found in the LICENSE file in the root directory of this source tree.

# usage: python glm_embed.py -d <batched_data_dir> -m glm.bin -b 1000 -o <output_dir>

import torch
from torch import nn
from meetneighbors.predictvfs.glm.gLM import *
from meetneighbors.predictvfs.glm.glm_utils import *
from tqdm import tqdm
import os
import sys
import numpy as np
import argparse
import pathlib
import datetime
import logging
import pickle as pk


def get_original_prot_ids(ids, id_dict):
    ori_ids = []
    for i in ids:
        if i != 0:
            if i not in id_dict.keys():
                ori_ids.append(str(i))
            else:
                ori_ids.append(id_dict[i])
        
    return ori_ids

def infer(logging, data_dir, model,output_path, device, id_dict, B_SIZE):
    f_list = os.listdir(data_dir)
    test_pkls=[]
    for pkl_f in f_list:
        if pkl_f == "prot_index_dict.pkl": 
            id_dict = pk.load(open(os.path.join(data_dir,pkl_f), "rb"))
            logging.info("found prot_index_dict.pkl file, using this as id mapping")
        else:
            test_pkls.append(str(os.path.join(data_dir,pkl_f)))
    logging.info(str(len(test_pkls))+" batched data pkl files to embed")
    
    torch.cuda.empty_cache()
    scaler = None
    HALF = True
    if HALF:
        logging.info("Inference with mixed precision model")
        scaler = torch.cuda.amp.GradScaler()
    
    best_layer = 1

    for pkl_f in tqdm(test_pkls, total=len(test_pkls)):
        input_embs = []
        hidden_embs = []
        hidden_embs2 = []
        label_embs = []
        all_contacts = []
        all_prot_ids = []
        predicted_embeds_masked = []
        all_probs = []
        output_embs=[]
        pickle_file =  open(pkl_f, 'rb')

        dataset = pk.load(pickle_file)
       
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=False)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False)
        for batch in loader:            
            inputs_embeds= batch['embeds'].type(torch.FloatTensor)        
            attention_mask = batch['attention_mask'].type(torch.FloatTensor)
            mask = torch.zeros(attention_mask.shape) #nothing is masked
            masked_tokens = (mask==1) & (attention_mask != 0)
            masked_tokens = torch.unsqueeze(masked_tokens, -1)
            masked_tokens = masked_tokens.to(device)
            inputs_embeds = inputs_embeds.to(device)
            inputs_embeds  = torch.where(masked_tokens, -1.0, inputs_embeds)
            attention_mask = attention_mask.to(device)
            labels = batch['label_embeds'].type(torch.FloatTensor)
            labels = labels.to(device)
            input_embs.append(inputs_embeds.cpu().detach().numpy())
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # call model
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False, output_hidden_states=True)
                    label_embs.append(labels.cpu().detach().numpy().astype(np.float16))
                    last_hidden_states = outputs.last_hidden_state
                    hidden_embs2.append(last_hidden_states.cpu().detach().numpy().astype(np.float16))
                    all_hidden_states = outputs.all_hidden_states
                    # print(all_hidden_states,flush=True)
                    hidden_embs.append(all_hidden_states.cpu().detach().numpy().astype(np.float16))

                         
                    prot_ids = batch['prot_ids']
                    all_contacts.append(outputs.contacts.cpu().detach().numpy().astype(np.float16))
                    all_prot_ids.append(prot_ids)
                    logits_all_preds = outputs.logits_all_preds
                    all_preds = logits_all_preds[masked_tokens.squeeze(-1)]
                    output_embs.append(logits_all_preds.cpu().detach().numpy().astype(np.float16))
                    predicted_probs = outputs.probs
                    raw_probs = predicted_probs.view(-1,4)
                    softmax = nn.Softmax(dim=1)
                    probs = softmax(raw_probs)
                    predicted_embeds_masked.append(all_preds.cpu().detach().numpy().astype(np.float16))
                    all_probs.append(probs.cpu().detach().numpy().astype(np.float16))
        input_embs = np.concatenate(np.concatenate(input_embs, axis = 0), axis = 0) # remove batch dimension
        hidden_embs2 =  np.concatenate(np.concatenate(hidden_embs2, axis = 0), axis = 0) # remove batch dimension
        print("OG embeds:",hidden_embs2.shape)
        hidden_embs = np.concatenate(hidden_embs, axis = 0)
        hidden_embs = hidden_embs.transpose(0,2,1,3) # change from (num_seqs, num_layers, seq_len, hidden_dim) to (num_seqs, seq_len, num_layers, hidden_dim)
        num_seqs, seq_len, num_layers, hidden_dim = hidden_embs.shape
        hidden_embs = hidden_embs.reshape(num_seqs * seq_len, num_layers, hidden_dim)
        print("New embeds:",hidden_embs.shape)
        assert torch.allclose(torch.tensor(hidden_embs[-1,-1,:]),torch.tensor(hidden_embs2[-1,:])), f"OG Embeding of shape {hidden_embs2.shape} and New Embedding of shape {hidden_embs.shape} did not match. Here are the embeds with OG first: {hidden_embs2[-1,:]} \n ===================== {hidden_embs[-1,-1,:]}"
        hidden_embs = hidden_embs[:,best_layer,:]
        print(f"New embeds after subset for best prediction layer {best_layer}:",hidden_embs.shape)
        # hidden_embs = np.concatenate(hidden_embs, axis = 0)
        label_embs =  np.concatenate(np.concatenate(label_embs, axis = 0), axis = 0) # remove batch dimension
        output_embs = np.concatenate(output_embs, axis = 0)

        x,y,z,_ = output_embs.shape 
        output_embs = np.concatenate(output_embs, axis = 0) # remove batch dimension
        all_probs = np.concatenate(all_probs, axis = 0)
        all_probs = all_probs.reshape(x,y,z)
        all_probs =  np.concatenate(all_probs, axis = 0) # remove batch dimension
        all_prot_ids = np.concatenate(np.concatenate(all_prot_ids, axis = 0), axis = 0)
        all_contacts = np.concatenate(all_contacts, axis =0)
        if id_dict != None:
            ori_prot_ids = get_original_prot_ids(all_prot_ids,id_dict)
        else:
            ori_prot_ids = all_prot_ids
        # remove padding
        label_embs = label_embs[np.where(all_prot_ids != 0)[0]]
        input_embs = input_embs[np.where(all_prot_ids != 0)[0]]
        hidden_embs = hidden_embs[np.where(all_prot_ids != 0)[0]]
        output_embs = output_embs[np.where(all_prot_ids != 0)[0]]
        all_probs = all_probs[np.where(all_prot_ids != 0)[0]]
        ALL_RESULTS,ATTENTION = False,False
        if ALL_RESULTS:
            results_filename = output_path+os.path.basename(pkl_f)+".results.pkl"
            results = {}
            results['label_embs'] = label_embs
            results['plm_embs'] = input_embs
            results['glm_embs'] = hidden_embs
            results['all_prot_ids'] = ori_prot_ids
            results['output_embs'] = output_embs
            results['all_probs'] = all_probs
            results_f = open(results_filename, "wb")
            pk.dump(results, results_f)
            results_f.close()
        if ATTENTION:
            attention_filename = output_path+os.path.basename(pkl_f)+".attention.pkl"
            attention_f = open(attention_filename, "wb")
            pk.dump(all_contacts,attention_f)
            attention_f.close()

        glm_embs = []

        for i,emb in enumerate(hidden_embs):
            glm_embs.append((ori_prot_ids[i],emb))
        glm_embs_filename = output_path+os.path.basename(pkl_f)+".glm.embs.pkl"
        embs_f = open(glm_embs_filename, "wb")
        pk.dump(glm_embs, embs_f)

        pickle_file.close()
    return None

# parser = argparse.ArgumentParser(description = "outputs glm embeddings")
# parser.add_argument('-d','--data_dir', type=pathlib.Path, help='batched data directory')
# parser.add_argument('-id', '--id_path',  help='path to prot_index_dict.pkl file', default = None)
# parser.add_argument('-m','--model_path', help="path to pretrained model, glm.bin")
# parser.add_argument('-b','--batch_size', type=int, help='batch_size', default = 100)
# parser.add_argument('-o', '--output_path', type=str, help='inference output directory', default = None)
# parser.add_argument('--attention',action='store_true', help='output attention matrices ', default = False)
# parser.add_argument('--hidden_size', type=int, help='hidden size', default = 1280)
# parser.add_argument('-n', '--ngpus', type=int, help='number of GPUs to use',  default = 1)
# parser.add_argument('-a', '--all_results',action='store_true', help='output all results including plm_embs/glm_embs/prot_ids/outputs/output_probabilitess', default = False)

# # load all arguments 
# args = parser.parse_args()
# if args.data_dir is None :
#     parser.error('--data_dir must be specified')
# if args.model_path is None :
#     parser.error('--model must be specified')

def run_glm_embeds(model,pkg_data_dir,glm_embed_output_path,device,ngpus,batch_size):
    # define all parameters
    data_dir = pkg_data_dir
    # model_path = args.model_path
    output_path = glm_embed_output_path
    # id_path = protidpkl_path
    B_SIZE = batch_size # batch size


    # make output folder if not specified
    # e = datetime.datetime.now()
    # if output_path == None:
    #     output_path = "glm_inference"+ e.strftime("-%d-%m-%Y-%H:%M:%S")
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    logfile_path = output_path+"/info.log"

    if not os.path.exists(output_path+"/results"):
        os.mkdir(output_path+"/results")
    results_dir = output_path+"/results/"

    # log is stored in both logfile and streamed to stdout
    # begin logging 

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[
            logging.FileHandler(logfile_path),
            logging.StreamHandler()])
    logging.info("output folder: " +output_path)
    logging.info("log file is located here: " +logfile_path)
    string_of_command = f"{' '.join(sys.argv)}"
    logging.info("command: " + string_of_command)

    model.eval()
    # ngpus=1 # I think it might be faster using torch parallel
    if ngpus>1:
        model = torch.nn.DataParallel(model)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info("batch_size: "+str(B_SIZE))

    # if id_path != None:
    #     id_dict = pk.load(open(id_path, "rb"))
    # else:
    id_dict = None
    with torch.no_grad():
        infer(logging,data_dir,model,output_path=results_dir,device=device, id_dict=id_dict,B_SIZE=B_SIZE) 
    return
