import gzip
import shutil

def compress(infile, tofile):
    with open(infile, 'rb') as f_in:
        with gzip.open(tofile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def decompress(infile, tofile):
    with gzip.open(infile, 'rb') as f_in:
        with open(tofile, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
