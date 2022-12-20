import os
import requests
import requests_random_user_agent
from multiprocessing import Pool

def download_files():
    '''
    Read filepaths and links from text file and pass them as tuple
    to a multiprocessing pool for downloading
    '''
    paths = []
    links = []

    # create the folder structure before downloading anything
    with open('links_and_paths.txt', 'r') as f:
        lines = f.readlines()
        for l in lines:
            split = l.split(',h')
            path, link = split
            path_split = path.replace(base, '').split("/")[:-1]
            for s in path_split:
                if not os.path.exists(s):
                    os.mkdir(s)
                os.chdir(s)
            os.chdir(base)
            links.append('h' + link)
            paths.append(path)
            
    # multiprocessing approach
    with Pool(processes=8) as p:
        with tqdm(total=len(links)) as pbar:
            for _ in p.imap_unordered(fetch, zip(links, paths)):
                pbar.update()
            
def fetch(tup):
    '''
    Function to pass to multiprocessing pool for parallel
    file downloads
    '''
    # no need to start a session to use random-user-agents
    # just import the library and it automatically assigns a user agent
    url, filepath = tup
    if not os.path.exists(filepath):
        r = requests.get(url)
        if r.status_code == 200:
            content = r.content
            # write the file
            with open(filepath, 'wb') as f:
                f.write(content)
        else:
            log.append(tup)

if __name__ == '__main__':
    base = os.getcwd()
    os.chdir(base)
    log = [] # no errors occurred, so this list was always empty
    download_files()
