from yt_dlp import YoutubeDL
import ffmpeg
from loguru import logger
import os
# from uuid import uuid4
import pandas as pd
import numpy as np
import uuid

def download_audio(link, output_dir):
    with YoutubeDL({
        'extract_audio': True,
        'format': 'bestaudio',
        'restrictfilenames': True,
        'trim_file_name': 50,
        'P': {output_dir: 'link'}
    }) as video:
        cwd = os.getcwd()
        os.chdir(output_dir)
        info_dict = video.extract_info(link, download = True)
        video_path = video.prepare_filename(info_dict)
        out_path = os.path.join(output_dir, video_path)

        logger.info(f"downloading: {out_path}")
        video.download(link)
        os.chdir(cwd)

        return out_path


def convert_audio(fn, out_fn):
    out, _ = (
        ffmpeg.input(fn)
        .output(out_fn, acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run()
    )

    print(out)


def download_audio_as_wav(url, output_dir):
    video_fn = download_audio(url, output_dir)
    out_fn = video_fn.replace('.webm', '.wav')

    convert_audio(video_fn, out_fn)
    os.unlink(video_fn)

    return out_fn


def _bin_by_sentences(transcript_df, window_size = 5):
    n_sentences = len(transcript_df)
    end = window_size
    bin_uuids = []
    while end < n_sentences+window_size:
        window_id = uuid.uuid4()
        bin_uuids.extend([window_id]*window_size)
        end+=window_size
    
    return bin_uuids[:n_sentences]

def _bin_by_time(transcript_df, window_size = 300):
    max_end = transcript_df['end'].max()+window_size
    time_bins = np.arange(0,transcript_df['end'].max()+window_size, window_size)
    binned_sentences = pd.cut(transcript_df['end'], time_bins)
    right_interval = binned_sentences.apply(lambda x: x.right)
    uuid_map = {right_val:uuid.uuid4() for right_val in right_interval.drop_duplicates()}
    bin_uuids = right_interval.map(uuid_map)
    return bin_uuids

def bin_sentences(transcript_df, by, window_size):
    if by == 'time':
        return _bin_by_time(transcript_df, window_size)
    elif by == 'sentences':
        return _bin_by_sentences(transcript_df, window_size)
    else:
        raise ValueError('currently only ["time", "sentences"] are supported')
        
def concat_segment_sents(sentences):
    agg_sent = {}
    agg_sent['text'] = ''.join(sentences['text'].tolist())
    agg_sent['start'] = sentences['start'].values[0]
    agg_sent['end'] = sentences['end'].values[-1]
    
    seg_char_counter = 0
    for _, sent in sentences.iterrows():
        for token in sent['tokens']:
            token['start_char'] += seg_char_counter
        # import pdb;pdb.set_trace()
        seg_char_counter += (len(sent['text']))
        
    agg_sent['tokens'] = np.concatenate(sentences['tokens'].tolist()).tolist()
    
    return pd.DataFrame.from_dict(agg_sent, orient = 'index').T

def chunk_transcript_segments(transcript_df, chunk_by, window_size):
    uuids = bin_sentences(transcript_df, chunk_by, window_size)
    transcript_df['segment_id'] = uuids
    segment_groups = transcript_df.groupby('segment_id')
    df_combined_sentences = segment_groups.apply(concat_segment_sents).sort_values('start')
    
    return df_combined_sentences

def _get_keyphrase_time_from_seg_tokens(kp, seg_tokens):
    
    for token in seg_tokens:
        if (token['start_char']+1) == kp['start']:
            kp['start_time'] = token['time']
    
    return kp

def get_segment_keyphrases(segmented_transcript, model):
    keyphrases = []
    for seg_id, segment in segmented_transcript.iterrows():
        text_keyphrases = model(segment['text'])
        for kp in text_keyphrases:
            kp['segment_id'] = seg_id[0]
            
            seg_tokens = segmented_transcript.loc[seg_id[0], 'tokens'].values[0]
            # import pdb;pdb.set_trace()
            kp = _get_keyphrase_time_from_seg_tokens(kp, seg_tokens)
            
            keyphrases.append(kp)
            
    return keyphrases

            
        
    return keyphrases