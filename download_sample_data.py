import os
import urllib.request
from urllib.parse import urlparse
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_data(url):
    target_dir = './data_temp'    
    os.makedirs(target_dir, exist_ok=True)    

    if 'dl=0' in url:
        url = url.replace('dl=0', 'dl=1')
    elif 'dl=1' not in url:
        url += '&dl=1' if '?' in url else '?dl=1'

    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_data.bin"
        
    output_path = os.path.join(target_dir, filename)
    
    print(f"Downloading '{filename}' to '{target_dir}' directory...")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
            
        print(f"Download completed successfully! Saved as: {output_path}\n")
    except Exception as e:
        print(f"Download failed. Error: {e}\n")

if __name__ == "__main__":
    print('Downloading sample 4D-STEM data from Dropbox...\n')
    
    original_file_url = 'https://www.dropbox.com/scl/fi/q7ss3sifnp1jaj0scfsvb/data.npy?rlkey=ugbslwzugve3dzz4jz2z0dql3&st=q25rj3wm&dl=1'
    recon_file_url = 'https://www.dropbox.com/scl/fi/x9givu71cvufayo0zk9yo/recon_cr_27.8.mgr?rlkey=1wwei1k4mlkywn9gq6at2hjfl&st=ndaexy5t&dl=1' 
    
    download_data(original_file_url)
    download_data(recon_file_url)