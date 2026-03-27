import os
import argparse
import gdown

def download_data(file_id):
    target_dir = './data'    
    os.makedirs(target_dir, exist_ok=True)    

    print(f"Downloading sample data to '{target_dir}' directory...")
    
    original_cwd = os.getcwd()
    
    try:
        os.chdir(target_dir)        
        downloaded_file = gdown.download(id=file_id, output=None, quiet=False)
        
        if downloaded_file:
            print(f"\nDownload completed successfully! Saved as: {target_dir}/{downloaded_file}")
        else:
            print("\nDownload failed. Please check the file ID and permissions.")
            
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download sample 4D-STEM data from Google Drive.")
    
    # TODO: 구글 드라이브 공유 링크에서 추출한 파일 ID를 아래에 입력하세요.
    DEFAULT_FILE_ID = '여기에_파일_ID를_입력하세요' 

    parser.add_argument('--id', type=str, default=DEFAULT_FILE_ID, help='Google Drive File ID')
    
    args = parser.parse_args()
    
    download_data(args.id)