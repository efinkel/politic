import whispercpp as w
from whispercpp import utils
from glob import glob
import os
import pandas as pd

api = utils.LazyLoader("api", globals(), "whispercpp.api_cpp2py_export")

def get_model(model_name):
    params = (
        api.Params
        .from_enum(api.SAMPLING_GREEDY)
        .with_translate(False)
        .with_print_realtime(False)
        .with_print_timestamps(True)
        .with_token_timestamps(True)
        .build()
    )

    model = w.Whisper.from_params(
        model_name,
        params
    ) 
    
    return model, params

def get_segment_token_data(seg_ind, model):
    token_data = []
    n_tokens = model.context.full_n_tokens(seg_ind)
    for token_ind in range(n_tokens):
        tok_text = model.context.full_get_token_text(seg_ind, token_ind)
        if (tok_text == '[_BEG_]') or ('_TT_' in tok_text):
            continue
        else:
            tok_metadata = model.context.full_get_token_data(seg_ind, token_ind)
            token_data.append({'token':tok_text, 'time':tok_metadata.t0})
        
    return token_data

def transcribe_w_time(model, params, fn):
    
    if isinstance(model, str):
        model, params = get_model()
        
    model.context.full(params, api.load_wav_file(fn).mono)
    
    results = []
    n_segments = model.context.full_n_segments()
    for seg_ind in range(n_segments):
        
        segment_result = {
            'start': model.context.full_get_segment_start(seg_ind)/100,
            'end': model.context.full_get_segment_end(seg_ind)/100,
            'text': model.context.full_get_segment_text(seg_ind)
        }
        
        segment_result['tokens'] = get_segment_token_data(seg_ind, model)
            
        results.append(segment_result)
        
    return results
    