#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 ETRI (Minkyu Lee)
import json
import logging
import timeit
import torch
import random
import os
import numpy as np
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets
from libs.base import fbank,logfbank,cmvn,cmvn2,set_cmvn_file
from espnet.asr.asr_utils import parse_hypothesis
from sklearn.preprocessing import normalize
from libs.feature_extraction.compute_fbank_feats import *
from libs.feature_extraction.compute_pitch_feats import compute_kaldi_pitch
from libs.feature_extraction.process_pitch_feats import process_pitch

import matplotlib.pyplot as plt
import pdb
from collections import OrderedDict
import itertools
import signal
import configargparse
def get_parser():
    parser = configargparse.ArgumentParser(
        description='Transcribe text from speech using a speech recognition model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--mode', type=str, default="cpu",choices=("cpu","gpu"),
                        help='CPU/GPU mode selection')
    parser.add('--dtype', choices=("float16", "float32", "float64"), default="float32",
                        help='Float precision')
    parser.add('--backend', type=str, default='pytorch',
                        choices=['chainer', 'pytorch'],
                        help='Backend library')
    parser.add('--model', type=str, 
                        help='음성인식모델')    
    parser.add('--use-vad', type=bool,help='vad 사용여부',choices=(True,False),default=False)
    parser.add('--beam-size', type=int, default=3,
                        help='탐색공간 후보 크기')
    parser.add('--rnnlm', type=str, default=None, #Optional
                        help='언어모델')
    parser.add('--lm-weight', type=float, default=0.1, #Optional
                        help='언어모델 가중치')    
    parser.add('--spm-mdl', type=str, default=None,
                        help='인식단위 모델')
    parser.add('--ctc-weight', type=float, default=0.3,
                        help='CTC weight')
    parser.add('--ncpu', type=int, default=1,help='단일채널 당 CPU thread수')
    parser.add('--ngpu', type=int, default=0,help='가용GPU')
    parser.add('--fast-decode',type=bool, default=False)
    parser.add('--filename', type=str, default=None)    
    return parser

class SpeechRecognizer:
    def __init__(self,args_in):
        self.args=args_in
        self.init(args_in)

    def init(self,args_in):

        self.args=args_in
        self.dim=83
        self.cmvn,self.dim=set_cmvn_file(os.path.dirname(self.args.model))        
        #print(self.cmvn)
        self.fb_buffer = np.empty((0,self.dim))
        args=args_in
        self.fast_decode=args.fast_decode 
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = (
            False  # https://github.com/pytorch/pytorch/issues/6351
        )
        

        model, train_args = load_trained_model(args.model)
        if self.args.ngpu==0:
            model=torch.quantization.quantize_dynamic(model)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        #train_args.preprocess_conf = None
        assert isinstance(model, ASRInterface)
        
        model.eval()
        load_inputs_and_targets = LoadInputsAndTargets(
            mode='asr', load_output=False, sort_in_input_length=False,
            preprocess_conf=None,
            preprocess_args={'train': False})
        self.lm = None 
        scorers = model.scorers()
        scorers["lm"] = self.lm
        scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
        weights = dict(
            decoder=1.0 - args.ctc_weight,
            ctc=args.ctc_weight,
            lm=args.lm_weight,
            length_bonus=0)
        beam_search = BeamSearch(
            beam_size=args.beam_size,
            vocab_size=len(train_args.char_list),
            weights=weights,
            scorers=scorers,
            sos=model.sos,
            eos=model.eos,
            token_list=train_args.char_list,
        )

        if args.ngpu > 1:
            raise NotImplementedError("only single GPU decoding is supported")
        if args.ngpu == 1:
            device = "cuda"
        else:
            device = "cpu"
            print("cpu mode")        
            torch.set_num_threads(4)
        
        dtype = getattr(torch, args.dtype)
        model.to(device=device, dtype=dtype).eval()
        beam_search.to(device=device, dtype=dtype).eval()
        self.train_args = train_args        
        self.load_inputs_and_targets = load_inputs_and_targets
        self.weights = weights        
        self.model = model
        self.beam_search = beam_search
        self.device = device
        self.dtype = dtype
        self.args = args 
        self.char_to_tokenid =  { train_args.char_list[i]:i  for i in range(0 , len(train_args.char_list) ) }
        self.spm_mdl = args.spm_mdl
        fbank_opts = FbankOptions()
        fbank_opts.register()
        fbank = Fbank(fbank_opts)
        self.fe = fbank
        self.start = timeit.default_timer()
        


    def recog_buffer(self,buffer,timeoutseconds=0):
        self.abort_flag=False
        self.start = timeit.default_timer()
        logging.info('Processing Start')     
        args = self.args
        train_args = self.train_args
        self.model.eval()
        
        self.m_buffer=buffer

        self.fb80=self.fe.compute_features(self.m_buffer,16000,1.0)
        self.fb80=cmvn2(self.fb80,in_norm=self.cmvn)

        self.fb_enc=np.zeros([self.fb80.shape[0],self.fb80.shape[1]+3])

        self.fb_enc[:,:80]=self.fb80
        #plt.imshow(self.fb_enc)
        #plt.savefig('fig2.png',dpi=300)
   
        with torch.no_grad():
                if self.args.ngpu==0: 
                            torch.set_num_threads(self.args.ncpu)           
                if not self.abort_flag:
                        self.enc = self.model.encode(torch.as_tensor(self.fb_enc).to(device=self.device, dtype=self.dtype))
                        self.enctime = timeit.default_timer()
                if not self.abort_flag:                                            
                        if self.fast_decode:
                            enc_output = np.argmax(self.model.ctc.log_softmax(self.enc.unsqueeze(0)).squeeze(0).cpu().detach().numpy(),axis=1)
                            ret =""
                            self.ctc_len=0
                            for idx in enc_output:
                                if self.train_args.char_list[idx] != '<blank>':
                                    ret += (self.train_args.char_list[idx])
                                    self.ctc_len+=1
                            ret=''.join(i for i, _ in itertools.groupby(list(ret)))
                            self.rec_text=ret.strip()
                        else:
                            self.nbest_hyps = self.beam_search(x=self.enc, maxlenratio=0, minlenratio=0)

                if self.fast_decode:
                    self.stop = timeit.default_timer()
                    logging.info('Processing Done : enc=%.2f dec=%.2f tot=%.2f rtf=%.2f'%((self.enctime-self.start),(self.stop-self.enctime),(self.stop-self.start),(self.stop-self.start)/(self.fb80.shape[0]/100)) )    
                    return self.rec_text.replace('▁+','').replace('▁',' ').replace('<eos>','')
                if self.abort_flag:
                    raise Excpetion('Error:abort() is called.')
                if self.nbest_hyps is None:
                    raise Exception('ERROR:Time(%.2f) is over'%timeoutseconds)
                self.nbest_hyps = [h.asdict() for h in self.nbest_hyps[:min(len(self.nbest_hyps), 1)]]
                self.stop = timeit.default_timer()
                
                logging.info('Processing Done : enc=%.2f dec=%.2f tot=%.2f rtf=%.2f'%((self.enctime-self.start),(self.stop-self.enctime),(self.stop-self.start),(self.stop-self.start)/(self.fb80.shape[0]/100)) )
                for self.n, self.hyp in enumerate(self.nbest_hyps, 1):
                    self.rec_text, self.rec_token, self.rec_tokenid, self.score = parse_hypothesis(self.hyp, train_args.char_list)

        return self.rec_text.replace('▁+','').replace('▁',' ').replace('<eos>','')
