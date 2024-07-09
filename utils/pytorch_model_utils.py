import os
import gzip
import torch
import shutil

def save_state_dict_to_gz(torch_object,
                          filename):
    '''
    Save state dict to gz file
    @param torch_object: model or optimizer, with state dict to save
    @param filename: str, path to save file, ends in .gz
    @return: None
    '''
    assert filename[-3:] == '.gz'
    torch.save(torch_object.state_dict(), filename[:-3])
    with open(filename[:-3], 'rb') as f_in, gzip.open(filename, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(filename[:-3])
    
def load_state_dict_from_gz(torch_object,
                            filename,
                            remove_uncompressed_file = True):
    '''
    Load state dict from gz file
    @param torch_object: model or optimizer, will modify state dict and return
    @param filename: str, path to file to load, ends in .gz
    @param remove_uncompressed_file: bool, whether to remove uncompressed .pt file, usually can set to True 
                                     unless expecting multiple threads to load the same model at the same time
    @return: torch_object, with loaded state dict
    '''
    assert filename[-3:] == '.gz'
    if not os.path.exists(filename[:-3]):
        with gzip.open(filename, 'rb') as f_in, open(filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    torch_object.load_state_dict(torch.load(filename[:-3]))
    if remove_uncompressed_file:
        os.remove(filename[:-3])
    return torch_object