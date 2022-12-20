import os
import time
import requests
# change user agent to avoid being banned by the server
import requests_random_user_agent 
from bs4 import BeautifulSoup

def writemode(fname):
    if fname not in os.listdir():
        mode = 'w'
    else:
        mode = 'a'
    return mode

def get_folders(link, path_so_far):
    '''
    Loop through all folders on the page, and if a folder is found, 
    call the function recursively and build up the path_so_far.

    If only files are found, save the built-up filepaths to a txt file
    along with the link for downloading later.

    Save any bad links or responses to a separate txt file
    '''
    page = requests.get(link)
    status = page.status_code
    if status == 200:
        print("Successful response")
        content = page.text
        soup = BeautifulSoup(content, 'lxml')
        all_links = soup.find_all('a')
        
        # find folders and files on page
        folder_links = [l['href'] for l in all_links \
                        if l.find_all('div', class_='folder')]
        file_links = [l['href'] for l in all_links \
            if l.find_all('div', class_='icon')]
            
        # if folders are present, there are no files
        if folder_links:
            folder_names = [l.find_all('div', class_='filename')[0]['title']\
                            for l in all_links if l.find_all(
                                    'div', class_='folder'
                                    )]
            for f_link, fldrname in zip(folder_links, folder_names):
                newpath = os.path.join(path_so_far, fldrname)
                # recursion
                get_folders(f_link, newpath)
                
        elif file_links:
            # if there are no folders, then there should be files
            savenames = [l.find_all('div', class_='filename')[0]['title']\
                         for l in all_links if l.find_all(
                                 'div', class_='icon'
                                 )]
            savepaths = [os.path.join(path_so_far, s) for s in savenames]
            
            # write links and paths to text file
            mode = writemode('links_and_paths.txt')
            with open('links_and_paths.txt', mode) as f:
                for filelink, filename in zip(file_links, savepaths):
                    f.write(f"{filename},{filelink}\n")
                    
            files.append(len(file_links))
            print(f"Found {len(file_links)} so far")
        else:
            print("No folders or files found at this link")
            
    else:
        mode = writemode('failed_links.txt')
        with open('failed_links.txt', mode) as f:
            f.write(link + '\n')
            f.write(f"Bad status code: {status}\n")

if __name__ == '__main__':
    url = input('Paste the base url to start scraping filepaths')
    base = os.getcwd()
    files = []

    start = time.time()
    get_folders(url, base)
    end = time.time()
    diff = end - start
    timestring = time.strftime("%H:%M:%S", time.gmtime(diff))
    
    print(f"Crawled folders and {sum(files)} files in {timestring}")
