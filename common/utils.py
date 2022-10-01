"""
Logger class files
"""
import sys
import torch
import random
import torch.nn as nn
from numpy import dot, zeros
from numpy.linalg import matrix_rank, norm


def get_model_info(id, lang=None):
    
    if id == 'ae_001':
        model_path  = 'model/vqvae/results/training/10000_instances/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/10000_instances/surf_vocab.json'   
    if id == 'ae_002':
        model_path  = 'model/vqvae/results/training/50000_instances/30epochs.pt'
        model_vocab = 'model/vqvae/results/training/50000_instances/surf_vocab.json'  
    
    if id == 'ae_finnish':
        model_path  = 'model/vqvae/results/training/finnish/55000_instances/15epochs.pt'
        model_vocab = 'model/vqvae/results/training/finnish/55000_instances/surf_vocab.json'  

    if id == 'ae_turkish':
        model_path  = 'model/vqvae/results/training/turkish/55204_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/turkish/55204_instances/surf_vocab.json'  


    if lang == 'finnish':    
        if id == 'batchsize128_beta_0.2_5x6_bi_kl_0.2_epc190':
            model_path  = 'model/vqvae/results/training/'+lang+'/55000_instances/batchsize128_beta0.2_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_190'
    
        if id == 'batchsize128_beta_0.2_6x6_bi_kl_0.2_epc190':
            model_path  = 'model/vqvae/results/training/'+lang+'/55000_instances/batchsize128_beta0.2_bi_kl0.2_6x6_dec256_suffixd300/200epochs.pt_190'
            model_vocab = 'model/vqvae/results/training/'+lang+'/55000_instances/batchsize128_beta0.2_bi_kl0.2_6x6_dec256_suffixd300/surf_vocab.json'  
    

    if lang == 'turkish':    
        if id == 'beta_0.1_5x6_bi_kl_0.2_epc190':
            model_path  = 'model/vqvae/results/training/55204_instances/beta0.1_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_190'
            model_vocab = 'model/vqvae/results/training/55204_instances/beta0.1_bi_kl0.2_5x6_dec256_suffixd300/surf_vocab.json'  
   
        if id == 'beta_0.5_5x6_bi_kl_0.2_epc160':
            model_path  = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.5_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_160'
            model_vocab = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.5_bi_kl0.2_5x6_dec256_suffixd300/surf_vocab.json'  
     
        if id == 'beta_0.2_5x6_bi_kl_0.2_epc170':
            model_path  = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.2_bi_kl0.2_5x6_dec256_suffixd300/200epochs.pt_170'
            model_vocab = 'model/vqvae/results/training/'+lang+'/55204_instances/beta0.2_bi_kl0.2_5x6_dec256_suffixd300/surf_vocab.json'  
     
        ## Semisup vqvae
        if id == 'semisup_batchsize128_beta_0.2_11x6_bi_kl_0.2_epc180':
            model_path = 'model/vqvae/results/training/turkish/semisup/55204_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/200epochs.pt_180'
            model_vocab = 'model/vqvae/results/training/turkish/semisup/55204_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append('model/vqvae/results/training/turkish/semisup/55204_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/'+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs
    
        if id == '12k_semisup_batchsize128_beta_0.2_11x6_bi_kl_0.2_epc180':
            model_path = 'model/vqvae/results/training/turkish/semisup/12798_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/200epochs.pt_180'
            model_vocab = 'model/vqvae/results/training/turkish/semisup/12798_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append('model/vqvae/results/training/turkish/semisup/12798_instances/batchsize128_beta0.2_bi_kl0.2_11x6_dec256_suffixd330/'+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs


        if id == 'late-supervision_batchsize128_beta_0.2_11x6_bi_kl_0.1_epc140':
            model_pfx = 'model/vqvae/results/training/turkish/late-supervision/55204_instances/batchsize128_beta0.2_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '300epochs.pt_140'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs
    

        if id == 'late-supervision_batchsize128_beta_0.1_11x6_bi_kl_0.1_epc120':
            model_pfx = 'model/vqvae/results/training/turkish/late-supervision/55204_instances/batchsize128_beta0.1_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '300epochs.pt_120'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs

        if id == 'early-supervision_batchsize128_beta_0.2_11x6_bi_kl_0.1_epc200':
            model_pfx = 'model/vqvae/results/training/turkish/early-supervision/12798_instances/batchsize128_beta0.2_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '301epochs.pt_15'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs

        if id == 'early-supervision_batchsize128_beta_0.2_11x6_bi_kl_0.1_epc200':
            model_pfx = 'model/vqvae/results/training/turkish/early-supervision/12798_instances/batchsize128_beta0.2_bi_kl0.1_11x6_dec256_suffixd660/'
            model_path  = model_pfx + '301epochs.pt_10'
            model_vocab = model_pfx + 'surf_vocab.json'
            tag_vocabs = []
            for i in range(11):
                tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
            return model_path, model_vocab, tag_vocabs
    

   


    #### SIGMORPHON 2016
    if id == 'ae_turkish_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/turkish/unsup/55000_instances/15epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/turkish/unsup/55000_instances/surf_vocab.json'  

    if id == 'ae_finnish_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/finnish/unsup/55000_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/finnish/unsup/55000_instances/surf_vocab.json'  
    

    if id == 'ae_finnish_unsup_1320':
        model_path  = 'model/vqvae/results/training/sig2016/finnish/unsup/55000_instances/55000_instances/27epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/finnish/unsup/55000_instances/55000_instances/surf_vocab.json'  

    if id == 'ae_hungarian_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/hungarian/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/hungarian/unsup/55000_instances/surf_vocab.json'  
    
    if id == 'ae_russian_unsup_1320':
        model_path  = 'model/vqvae/results/training/sig2016/russian/unsup/55000_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/russian/unsup/55000_instances/surf_vocab.json'  

    if id == 'ae_german_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/german/unsup/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/german/unsup/55000_instances/surf_vocab.json'  

    if id == 'ae_spanish_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/spanish/unsup/55000_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/spanish/unsup/55000_instances/surf_vocab.json'  
   
    if id == 'ae_georgian_unsup_640':
        model_path  = 'model/vqvae/results/training/sig2016/georgian/unsup/55000_instances/6epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/georgian/unsup/55000_instances/surf_vocab.json'  


    ### SIGMORPHON 2016- LATE SUPERVISION
    if id == 'turkish_late':
        model_pfx    = 'model/vqvae/results/training/sig2016/turkish/late-sup/42406_instances/run1_batchsize128_beta0.7_bi_kl0.1_11x10_dec512_suffixd660/'
        #model_pfx    = 'model/vqvae/results/training/sig2016/turkish/late-sup/42406_instances/batchsize128_beta0.7_bi_kl0.1_11x10_dec512_suffixd660/'
        model_path   = model_pfx + '200epochs.pt'
        model_vocab  = model_pfx + 'surf_vocab.json'  
        tag_vocabs = model_pfx + 'tag_vocabs.json'  
    
        return model_path, model_vocab, tag_vocabs


    if id == 'turkish_late_msved':
        model_pfx    = 'model/msved/results/training/late-sup/42406_instances/batchsize128_nz128_kl0.2_dec256/'
        model_path   = model_pfx + '150epochs.pt'
        model_vocab  = model_pfx + 'surf_vocab.json'  
        tag_vocabs = model_pfx + 'tag_vocabs.json'  
        return model_path, model_vocab, tag_vocabs


    if id == 'spanish_late':
        model_pfx    = 'model/vqvae/results/training/sig2016/spanish/late-sup/84950_instances/batchsize128_beta0.7_bi_kl0.1_11x10_dec512_suffixd660/'
        model_path   = model_pfx + '301epochs.pt'
        model_vocab  = model_pfx + 'surf_vocab.json'  
        tag_vocabs = model_pfx + 'tag_vocabs.json'  
        return model_path, model_vocab, tag_vocabs


    if id == 'finnish_late':
        model_pfx    = 'model/vqvae/results/training/sig2016/finnish/late-sup/87487_instances/batchsize128_beta0.5_bi_kl0.1_11x10_dec512_suffixd660/'
        model_path   = model_pfx + '200epochs.pt'
        model_vocab  = model_pfx + 'surf_vocab.json'  
        tag_vocabs = model_pfx + 'tag_vocabs.json'  
        return model_path, model_vocab, tag_vocabs





    if id == 'german_late':
        model_pfx    = 'model/vqvae/results/training/sig2016/german/late-sup/69023_instances/batchsize128_beta0.7_bi_kl0.1_11x10_dec512_suffixd660/'
        model_path   = model_pfx + '200epochs.pt'
        model_vocab  = model_pfx + 'surf_vocab.json'  
        tag_vocabs = []
        for i in range(11):
            tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
        return model_path, model_vocab, tag_vocabs

    if id == 'russian_late':
        model_pfx    = 'model/vqvae/results/training/sig2016/russian/late-sup/80489_instances/batchsize128_beta0.5_bi_kl0.1_12x10_dec512_suffixd1320/'
        model_path   = model_pfx + '200epochs.pt'
        model_vocab  = model_pfx + 'surf_vocab.json'  
        tag_vocabs = []
        for i in range(12):
            tag_vocabs.append(model_pfx+str(i)+'_tagvocab.json')  
        return model_path, model_vocab, tag_vocabs



    if id == 'ae_navajo_unsup_240':
        model_path  = 'model/vqvae/results/training/navajo/unsup/32109_instances/16epochs.pt'
        model_vocab = 'model/vqvae/results/training/navajo/unsup/32109_instances/surf_vocab.json'  
   
    if id == 'ae_navajo_unsup_360':
        model_path  = 'model/vqvae/results/training/navajo/sig2016/32109_instances/15epochs.pt'
        model_vocab = 'model/vqvae/results/training/navajo/sig2016/32109_instances/surf_vocab.json'  

    if id == 'sigmorphon2016_ae_arabic_gru_unsup_660':
        model_path  = 'model/vqvae/results/training/arabic/sig2016/gru/55000_instances/enc_nh_660/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/arabic/sig2016/gru/55000_instances/enc_nh_660/surf_vocab.json'  

    if id == 'sigmorphon2016_ae_arabic_gru_unsup_330':
        model_path  = 'model/vqvae/results/training/arabic/sig2016/gru/55000_instances/enc_nh_330/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/arabic/sig2016/gru/55000_instances/enc_nh_330/surf_vocab.json'  

    if id == 'sigmorphon2016_ae_maltese_gru_unsup_660':
        model_path  = 'model/vqvae/results/training/maltese/sig2016/gru/55000_instances/enc_nh_660/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/maltese/sig2016/gru/55000_instances/enc_nh_660/surf_vocab.json'  


    if id == 'ae_maltese_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/maltese/unsup/lstm/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/maltese/unsup/lstm/55000_instances/surf_vocab.json'  
   
    if id == 'ae_arabic_unsup_660':
        model_path  = 'model/vqvae/results/training/sig2016/arabic/unsup/lstm/55000_instances/17epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2016/arabic/unsup/lstm/55000_instances/surf_vocab.json'  



    #### SIGMORPHON2021
    if id == 'sigmorphon2021_ae_spa_unsup_360':
        model_path  = 'model/vqvae/results/training/spa/sig2021/55000_instances/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/spa/sig2021/55000_instances/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ara_gru_unsup_360':
        model_path  = 'model/vqvae/results/training/ara/sig2021/gru/55000_instances/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/ara/sig2021/gru/55000_instances/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_spa_gru_unsup_360':
        model_path  = 'model/vqvae/results/training/spa/sig2021/gru/55000_instances/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/spa/sig2021/gru/55000_instances/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_bul_gru_unsup_360':
        model_path  = 'model/vqvae/results/training/bul/sig2021/gru/39011_instances/enc_nh_360/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/bul/sig2021/gru/39011_instances/enc_nh_360/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_bul_gru_unsup_660':
        model_path  = 'model/vqvae/results/training/bul/sig2021/gru/39011_instances/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/bul/sig2021/gru/39011_instances/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_bul_gru_unsup_300':
        model_path  = 'model/vqvae/results/training/bul/sig2021/gru/39011_instances/enc_nh/300/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/bul/sig2021/gru/39011_instances/enc_nh/300/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ara_gru_unsup_660':
        model_path  = 'model/vqvae/results/training/ara/sig2021/gru/110000_instances/enc_nh_660/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/ara/sig2021/gru/110000_instances/enc_nh_660/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ara_gru_unsup_360':
        model_path  = 'model/vqvae/results/training/ara/sig2021/gru/110000_instances/enc_nh_360/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/ara/sig2021/gru/110000_instances/enc_nh_360/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ara_gru_unsup_300':
        model_path  = 'model/vqvae/results/training/ara/sig2021/gru/110000_instances/enc_nh_300/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/ara/sig2021/gru/110000_instances/enc_nh_300/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ind_gru_unsup_300':
        model_path  = 'model/vqvae/results/training/ind/sig2021/gru/11072_instances/enc_nh_300/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/ind/sig2021/gru/11072_instances/enc_nh_300/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ind_gru_unsup_660':
        model_path  = 'model/vqvae/results/training/ind/sig2021/gru/11072_instances/enc_nh_660/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/ind/sig2021/gru/11072_instances/enc_nh_660/surf_vocab.json'  


    if id == 'sigmorphon2021_ae_ame_gru_unsup_300':
        model_path  = 'model/vqvae/results/training/ame/sig2021/gru/2524_instances/enc_nh_300/35epochs.pt'
        model_vocab = 'model/vqvae/results/training/ame/sig2021/gru/2524_instances/enc_nh_300/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_ame_gru_unsup_658':
        model_path  = 'model/vqvae/results/training/ame/sig2021/gru/2524_instances/enc_nh_658/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/ame/sig2021/gru/2524_instances/enc_nh_658/surf_vocab.json'  


    if id == 'sigmorphon2021_ae_tur_gru_unsup_663':
        model_path  = 'model/vqvae/results/training/tur/sig2021/gru/55000_instances/enc_nh_663/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/tur/sig2021/gru/55000_instances/enc_nh_663/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_cni_gru_unsup_630':
        model_path  = 'model/vqvae/results/training/cni/sig2021/gru/13948_instances/enc_nh_630/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/cni/sig2021/gru/13948_instances/enc_nh_630/surf_vocab.json'  


    if id == 'sigmorphon2021_ae_krl_gru_unsup_660':
        model_path  = 'model/vqvae/results/training/krl/sig2021/gru/55000_instances/enc_nh_660/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/krl/sig2021/gru/55000_instances/enc_nh_660/surf_vocab.json'  
    

    if id == 'sigmorphon2021_ae_ind_lstm_unsup_660':
        model_path  = 'model/vqvae/results/training/ind/sig2021/lstm/11072_instances/enc_nh_660/20epochs.pt'
        model_vocab = 'model/vqvae/results/training/ind/sig2021/lstm/11072_instances/enc_nh_660/surf_vocab.json'  


    if id == 'sigmorphon2021_ae_tur_lstm_unsup_663':
        model_path  = 'model/vqvae/results/training/tur/sig2021/lstm/55000_instances/enc_nh_663/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/tur/sig2021/lstm/55000_instances/enc_nh_663/surf_vocab.json'  

    if id == 'sigmorphon2021_ae_krl_lstm_unsup_660':
        model_path  = 'model/vqvae/results/training/krl/sig2021/lstm/55000_instances/enc_nh_660/5epochs.pt'
        model_vocab = 'model/vqvae/results/training/krl/sig2021/lstm/55000_instances/enc_nh_660/surf_vocab.json'  

    #### MORPHOCHALLENGE - SEGMENTATION 
    if id == 'vae_segm_45':
        model_path  = 'model/vae/results/training/50000_instances/50epochs.pt_45'
        model_vocab = 'model/vae/results/training/50000_instances/surf_vocab.json'  

    if id == 'vae_segm_35':
        model_path  = 'model/vae/results/training/50000_instances/50epochs.pt_35'
        model_vocab = 'model/vae/results/training/50000_instances/surf_vocab.json'  

    if id == 'vae_segm_25':
        model_path  = 'model/vae/results/training/50000_instances/50epochs.pt_25'
        model_vocab = 'model/vae/results/training/50000_instances/surf_vocab.json'  

    if id == 'vae_segm_15':
        model_path  = 'model/vae/results/training/50000_instances/50epochs.pt_15'
        model_vocab = 'model/vae/results/training/50000_instances/surf_vocab.json'  

    if id == 'vae_segm_10':
        model_path  = 'model/vae/results/training/50000_instances/50epochs.pt_10'
        model_vocab = 'model/vae/results/training/50000_instances/surf_vocab.json'  

    if id == 'vae_segm_02_10':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_10'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  

    if id == 'vae_segm_02_15':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_15'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  

    if id == 'vae_segm_02_20':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_20'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  

    if id == 'vae_segm_02_25':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_25'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  
    
    if id == 'vae_segm_02_30':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_30'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  


    if id == 'vae_segm_02_35':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_35'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  

    if id == 'vae_segm_02_40':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0/50epochs.pt_40'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0/surf_vocab.json'  


    if id == 'vae_segm_04_20':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_0.2_warmup30_enc_nh256_decdout_in0.5/50epochs.pt_20'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_0.2_warmup30_enc_nh256_decdout_in0.5/surf_vocab.json'  
   
    if id == 'vae_segm_05_05':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/50epochs.pt_5'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/surf_vocab.json'  

    if id == 'vae_segm_05_10':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/50epochs.pt_10'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/surf_vocab.json'  

    if id == 'vae_segm_05_20':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/50epochs.pt_20'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/surf_vocab.json'  


    if id == 'vae_segm_05_25':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/50epochs.pt_25'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/surf_vocab.json'  


    if id == 'vae_segm_05_30':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/50epochs.pt_30'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup50_enc_nh256_decdout_in0.2/surf_vocab.json'  



    if id == 'vae_segm_06_40':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/50epochs.pt_40'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/surf_vocab.json'  

    if id == 'vae_segm_06_30':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/50epochs.pt_30'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/surf_vocab.json'  

    if id == 'vae_segm_06_20':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/50epochs.pt_20'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/surf_vocab.json'  

    if id == 'vae_segm_06_10':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/50epochs.pt_10'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh256_decdout_in0.2/surf_vocab.json'  



    if id == 'vae_segm_07_45':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.2_batchsize128_maxkl_0.2_warmup10_enc_nh256_decdout_in0.2/50epochs.pt_45'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.2_batchsize128_maxkl_0.2_warmup10_enc_nh256_decdout_in0.2/surf_vocab.json'  


    if id == 'vae_segm_08_10':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.3_nz32/51epochs.pt_10'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.3_nz32/surf_vocab.json'  


    if id == 'vae_segm_08_50':
        model_path  = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.3_nz32/51epochs.pt_50'
        model_vocab = 'model/vae/results/training/50000_instances/kl_start0.0_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.3_nz32/surf_vocab.json'  

    if id == 'vae_segm_09_15':
        model_path  = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/52534_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/51epochs.pt_50'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/52534_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/surf_vocab.json'  

    if id == 'vae_segm_09_50':
        model_path  = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/52534_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/51epochs.pt_15'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/52534_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/surf_vocab.json'  

    if id == 'neubig_vae_50k':
        model_path  = '/kuacc/users/mugekural/workfolder/dev/git/trmor/trmor_aggressive1_kls0.10_warm10_0_0_101.pt11'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/50000_surf_vocab.json'  

    if id == 'neubig_vae_50k_agg0_5':
        model_path= '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/models/trmor/trmor_aggressive0_kls0.10_warm10_0_0_101.pt5'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/50000_surf_vocab.json'

    if id == 'neubig_vae_50k_agg0_15':
        model_path= '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/models/trmor/trmor_aggressive0_kls0.10_warm10_0_0_101.pt15'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/50000_surf_vocab.json'

    if id == 'neubig_vae_50k_agg0_19':
        model_path= '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/models/trmor/trmor_aggressive0_kls0.10_warm10_0_0_101.pt19'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/50000_surf_vocab.json'


    if id == 'neubig_vae_617k_10':
        model_path= '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/models/trmor/trmor_aggressive1_kls0.10_warm10_0_0_202.pt10'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/vae-lagging-encoder/617298_surf_vocab.json'


    if id == 'VAE_FINAL':
        model_path = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/50000_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/51epochs.pt_50'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/50000_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/surf_vocab.json'


    if id == 'CHARLM_FINAL':
        model_path = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/charlm/results/training/50000_instances/for_segm/30epochs.pt'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/charlm/results/training/50000_instances/for_segm/surf_vocab.json'

    if id == 'CHARLM_FINAL_TEST':
        model_path = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/charlm/results/training/52534_instances/for_segm_filter_test/30epochs.pt'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/charlm/results/training/52534_instances/for_segm_filter_test/surf_vocab.json'



    if id == 'CHARLM_GPT_FINAL':
        model_path  = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/miniGPT/results/training/50000_instances/100epochs.pt'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/miniGPT/results/training/50000_instances/surf_vocab.json'


    if id == 'CHARLM_GPT_TEST':
        model_path  = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/miniGPT/results/training/52534_instances/filter_TEST100epochs.pt'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/miniGPT/results/training/52534_instances/filter_TEST/surf_vocab.json'


    #### SIGMORPHON2021- task2
    if id == 'turkish_ae_640':
        model_path  = 'model/vqvae/results/training/sig2021-task2/Turkish/unsup/616418_instances/6epochs.pt'
        model_vocab = 'model/vqvae/results/training/sig2021-task2/Turkish/unsup/616418_instances/surf_vocab.json'  


    ### THESIS- SIG2016
    if id == 'georgian_thesis_sig2016':
        model_path  = 'model/vqvae/results/training/sig2016/georgian/early-supervision-thesis/24000_instances/run1-batchsize128_beta0.1_bi_kl0.1_8x1_dec256_suffixd640/301epochs.pt_direct'
        model_vocab = 'model/vqvae/results/training/sig2016/georgian/early-supervision-thesis/24000_instances/run1-batchsize128_beta0.1_bi_kl0.1_8x1_dec256_suffixd640/surf_vocab.json'  
    if id == 'georgian_thesis_sig2016_withux':
        model_path  = 'model/vqvae/results/training/sig2016/georgian/early-supervision-thesis_withux/12795_instances/run1-batchsize128_beta0.1_bi_kl0.1_8x1_dec256_suffixd640/301epochs.pt_direct'
        model_vocab = 'model/vqvae/results/training/sig2016/georgian/early-supervision-thesis_withux/12795_instances/run1-batchsize128_beta0.1_bi_kl0.1_8x1_dec256_suffixd640/surf_vocab.json'  

    if id == 'german_thesis_sig2016':
        model_path  = 'model/vqvae/results/training/sig2016/german/early-supervision-thesis/12777_instances/run1-batchsize128_beta0.1_bi_kl0.1_11x1_dec512_suffixd660/301epochs.pt_direct'
        model_vocab = 'model/vqvae/results/training/sig2016/german/early-supervision-thesis/12777_instances/run1-batchsize128_beta0.1_bi_kl0.1_11x1_dec512_suffixd660/surf_vocab.json'  
    if id == 'german_thesis_sig2016_withux':
        model_path  = 'model/vqvae/results/training/sig2016/german/early-supervision-thesis_withux/12777_instances/run1-batchsize128_beta0.1_bi_kl0.1_11x1_dec512_suffixd660/301epochs.pt_direct'
        model_vocab = 'model/vqvae/results/training/sig2016/german/early-supervision-thesis_withux/12777_instances/run1-batchsize128_beta0.1_bi_kl0.1_11x1_dec512_suffixd660/surf_vocab.json'  


    if id == 'finnish_thesis_sig2016':
        model_path  = 'model/vqvae/results/training/sig2016/finnish/early-supervision-thesis/24000_instances/run1-batchsize128_beta0.1_bi_kl0.2_11x1_dec512_suffixd1320/301epochs.pt_direct'
        model_vocab = 'model/vqvae/results/training/sig2016/finnish/early-supervision-thesis/24000_instances/run1-batchsize128_beta0.1_bi_kl0.2_11x1_dec512_suffixd1320/surf_vocab.json'  
    if id == 'finnish_thesis_sig2016_withux':
        model_path  = 'model/vqvae/results/training/sig2016/finnish/early-supervision-thesis_withux/12800_instances/run1-batchsize128_beta0.1_bi_kl0.2_11x1_dec512_suffixd1320/301epochs.pt_direct'
        model_vocab = 'model/vqvae/results/training/sig2016/finnish/early-supervision-thesis_withux/12800_instances/run1-batchsize128_beta0.1_bi_kl0.2_11x1_dec512_suffixd1320/surf_vocab.json'  

    if id == 'vae_test':
        model_path  = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/4587_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/51epochs.pt_50'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vae/results/training/4587_instances/kl_start0.1_batchsize128_maxkl_1.0_warmup10_enc_nh512_decdout_in0.2_nz32/TEST/surf_vocab.json'  


    if id == 'hungarian_vqvae_probe':
        model_path = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vqvae/results/training/sig2016/hungarian/early-supervision-thesis_withux/19200_instances/run1-batchsize128_beta0.1_bi_kl0.2_10x1_dec256_suffixd660/301epochs.pt_direct'
        model_vocab = '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vqvae/results/training/sig2016/hungarian/early-supervision-thesis_withux/19200_instances/run1-batchsize128_beta0.1_bi_kl0.2_10x1_dec256_suffixd660/surf_vocab.json'
        tag_vocabs =  '/kuacc/users/mugekural/workfolder/dev/git/trmor/model/vqvae/results/training/sig2016/hungarian/early-supervision-thesis_withux/19200_instances/run1-batchsize128_beta0.1_bi_kl0.2_10x1_dec256_suffixd660/tag_vocabs.json'  
        return model_path, model_vocab, tag_vocabs

    return model_path, model_vocab

class Logger(object):
  def __init__(self, output_file):
    self.terminal = sys.stdout
    self.log = open(output_file, "w")

  def write(self, message):
    print(message, end="", file=self.terminal, flush=True)
    print(message, end="", file=self.log, flush=True)

  def flush(self):
    self.terminal.flush()
    self.log.flush()
 

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)


class xavier_normal_initializer(object):
    def __call__(self, tensor):
        nn.init.xavier_normal_(tensor)


def plot_curves(task, bmodel, fig, ax, trn_loss_values, val_loss_values, style, ylabel):
    ax.plot(range(len(trn_loss_values)), trn_loss_values, style, label=bmodel+'_trn')
    ax.plot(range(len(val_loss_values)), val_loss_values, style,label=bmodel+'_val')
    if ylabel != 'acc': # hack for clean picture
        leg = ax.legend() #(loc='upper right', bbox_to_anchor=(0.5, 1.35), ncol=3)
        ax.set_title(task,loc='left')
    if ylabel != 'loss':
        ax.set_xlabel('epochs')
    ax.set_ylabel(ylabel)       


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_li_vectors(dim, R):
    r = matrix_rank(R) 
    index = zeros( r ) #this will save the positions of the li columns in the matrix
    counter = 0
    index[0] = 0 #without loss of generality we pick the first column as linearly independent
    j = 0 #therefore the second index is simply 0

    for i in range(R.shape[1]): #loop over the columns
        if i != j: #if the two columns are not the same
            inner_product = dot( R[:,i], R[:,j] ) #compute the scalar product
            norm_i = norm(R[:,i]) #compute norms
            norm_j = norm(R[:,j])

            #inner product and the product of the norms are equal only if the two vectors are parallel
            #therefore we are looking for the ones which exhibit a difference which is bigger than a threshold
            if abs(inner_product - norm_j * norm_i) > 1e-4:
                counter += 1 #counter is incremented
                index[counter] = i #index is saved
                j = i #j is refreshed
            #do not forget to refresh j: otherwise you would compute only the vectors li with the first column!!
    R_independent = zeros((r, dim))
    i = 0
    #now save everything in a new matrix
    while( i < dim ):
        #R_independent[i,:] = R[index[i],:] 
        R_independent[:,i] = R[:,index[i]] 
        i += 1
    return R_independent


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


'''lines = []
with open('trn_4x10.txt', 'r') as reader:
    for line in reader:
        lines.append(line)

random.shuffle(lines)

with open('trn_4x10_shuffled.txt', 'w') as writer:
    for line in lines[:10000]:
        writer.write(line)

with open('val_4x10_shuffled.txt', 'w') as writer:
    for line in lines[10000:]:
        writer.write(line)'''