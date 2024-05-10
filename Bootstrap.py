import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile

os_name, os_release, os_version = platform.system(), platform.release(), platform.version()

MODEL_URL = "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf"
MODEL_FILENAME = "Phi-3-mini-4k-instruct-q4.gguf"
CURL_URL = "https://cosmo.zip/pub/cosmos/bin/curl"
CURL_FILENAME = "curl.com" if os_name == 'Windows' else "curl"
LLAMAFILE_INFO = "Mozilla-Ocho/llamafile"
LLAMAFILE_REGEX = r"llamafile-\d+\.\d+\.\d+"
LLAMAFILE_FILENAME = "llamafile.com" if os_name == 'Windows' else "llamafile"
AISERVER_INFO = "igoforth/RWAILib"
AISERVER_REGEX = r"AIServer\.zip"
AISERVER_FILENAME = "AIServer.zip"

def resolve_github(curl_path: pathlib.Path, repo: str, file_s: str) -> str:
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    download_url = ""
    file_r = re.compile(file_s)

    # get the latest release from the API
    response = download(curl_path, api_url)
    
    if not response:
        print(f"Failed to get a response from {api_url}")
        sys.exit(1)
    
    # parse the response
    try:
        response_json = json.loads(response)
        
        # get the first asset that matches the pattern
        for asset in response_json['assets']:
            if file_r.match(asset['name']):
                download_url = asset['browser_download_url']
                break
            
        if not download_url:
            print(f"Failed to find a download URL for {repo}")
            sys.exit(1)
        
    except Exception as e:
        print(f"Failed to parse the response from {api_url} due to {e}")
        sys.exit(1)
    
    return download_url

def run_cmd(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        print(f"Failed to run {command} due to {e}")
        sys.exit(1)

def download(curl_path: pathlib.Path, url: str, dst: pathlib.Path | None = None) -> str | None:

    # if no destination is provided, caller wants the contents of the file
    if not dst:
        destination = tempfile.NamedTemporaryFile(delete=False).name
    else:
        destination = str(dst)

    command = f"{str(curl_path)} -L {url} -o {destination}"

    run_cmd(command)

    if not dst:
        # return the contents of the file and delete the file
        with open(destination, 'r') as f:
            contents = f.read()
            
        # delete the file
        os.unlink(destination)
        
        return contents
    else:
        return None

def bootstrap(dst: pathlib.Path):
    models = dst / "models"
    bins = dst / "bin"
    
    # create the directories
    models.mkdir(parents=True, exist_ok=True)
    bins.mkdir(parents=True, exist_ok=True)
    
    # check for CURL
    if os_name == 'Windows' and int(os_version.split(".")[2]) < 17063:
        # Bootstrap CURL
        curl_path = bins / CURL_FILENAME
        command = f"powershell -Command \"Invoke-WebRequest {CURL_URL} -OutFile {curl_path}\""
        run_cmd(command)
    else:
        curl_path = shutil.which("curl")

    if not curl_path:
        print("Failed to find CURL")
        sys.exit(1)
    else:
        curl_path = pathlib.Path(curl_path)

    # download llamafile
    llamafile_url = resolve_github(curl_path, LLAMAFILE_INFO, LLAMAFILE_REGEX)
    llamafile_path = bins / LLAMAFILE_FILENAME
    download(curl_path, llamafile_url, llamafile_path)
    
    if os_name != 'Windows':
        # make llamafile executable
        os.chmod(llamafile_path, 0o755)

    # download the AI Server
    aiserver_url = resolve_github(curl_path, AISERVER_INFO, AISERVER_REGEX)
    aiserver_path = dst / AISERVER_FILENAME
    download(curl_path, aiserver_url, aiserver_path)

    # download the model
    model_path = models / MODEL_FILENAME
    download(curl_path, MODEL_URL, model_path)

    print("Bootstrap successful")

def main():
    
    # convert relative to absolute path
    directory = pathlib.Path(os.getcwd()).resolve()
    
    # bootstrap
    bootstrap(directory)
    
if __name__ == "__main__":
    main()