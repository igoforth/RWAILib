import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile

(
    os_name,
    os_release,
    os_version,
) = (
    platform.system(),
    platform.release(),
    platform.version(),
)

SHELL = "powershell.exe" if platform.system() == "Windows" else "sh"
MODEL_URL = "https://huggingface.co/PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed/resolve/main/Phi-3-mini-128k-instruct.Q4_K_M.gguf"
MODEL_FILENAME = "Phi-3-mini-128k-instruct.Q4_K_M.gguf"
CURL_URL = "https://cosmo.zip/pub/cosmos/bin/curl"
CURL_FILENAME = "curl.com" if os_name == "Windows" else "curl"
LLAMAFILE_INFO = "Mozilla-Ocho/llamafile"
LLAMAFILE_REGEX = r"llamafile-\d+\.\d+\.\d+"
LLAMAFILE_FILENAME = "llamafile.com" if os_name == "Windows" else "llamafile"
AISERVER_INFO = "igoforth/RWAILib"
AISERVER_REGEX = r"AIServer\.pyz"
AISERVER_FILENAME = "AIServer.pyz"
VERSION_FILENAME = ".version"


def get_github_version(curl_path: pathlib.Path, cwd: pathlib.Path, repo: str) -> str:
    api_url: str = f"https://api.github.com/repos/{repo}/releases/latest"
    user_agent: str = "igoforth/RWAILib"

    response = download(curl_path, cwd, api_url, user_agent=user_agent)

    if not response:
        print(
            "Failed to retrieve the latest release information. Exiting...",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        return json.loads(response)["tag_name"]
    except json.JSONDecodeError as e:
        print(
            f"Failed to parse the response from {api_url} due to {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def resolve_github(
    curl_path: pathlib.Path,
    cwd: pathlib.Path,
    repo: str,
    file_s: str,
    windows_fallback: bool,
) -> str:
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    user_agent: str = "igoforth/RWAILib"
    download_url = ""
    file_r = re.compile(file_s)

    # get the latest release from the API
    response = download(
        curl_path,
        cwd,
        api_url,
        windows_fallback=windows_fallback,
        user_agent=user_agent,
    )

    if not response:
        print(f"Failed to get a response from {api_url}", file=sys.stderr)
        sys.exit(1)

    # parse the response
    try:
        response_json = json.loads(response)

        # get the first asset that matches the pattern
        for asset in response_json["assets"]:
            if file_r.match(asset["name"]):
                download_url = asset["browser_download_url"]
                break

        if not download_url:
            print(f"Failed to find a download URL for {repo}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(
            f"Failed to parse the response from {api_url} due to {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    return download_url


def run_cmd(command: str) -> None:
    string_command = "`" + command + "`"
    try:
        print(string_command)
        subprocess.run(
            command,
            shell=True,
            executable=SHELL,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {string_command} due to {e}", file=sys.stderr)
        sys.exit(1)


def download(
    curl_path: pathlib.Path,
    cwd: pathlib.Path,
    url: str,
    dst: pathlib.Path | None = None,
    windows_fallback: bool = False,
    resume_flag: bool = False,
    user_agent: str = "",
) -> str | None:

    # if no destination is provided, caller wants the contents of the file
    if not dst:
        destination = str(
            pathlib.Path(tempfile.NamedTemporaryFile(delete=False).name).relative_to(
                cwd
            )
        )
    else:
        destination = str(dst)

    if not windows_fallback:
        if resume_flag:
            command = f'{str(curl_path)} -C - -ZL "{url}" -o "{destination}"'
        else:
            command = f'{str(curl_path)} -ZL "{url}" -o "{destination}"'
        if user_agent:
            command += f' -A "{user_agent}"'
    else:
        destination = str(pathlib.Path(destination).resolve())
        drv_ltr = (re.search(r"^/(\w)/.*$", destination)).group(1)  # type: ignore
        destination = drv_ltr + ":" + destination[2:].replace("/", "\\")
        command = f'bitsadmin /transfer 1 "{url}" "{destination}"\''

    run_cmd(command)

    if not dst:
        # return the contents of the file and delete the file
        with open(destination, "r") as f:
            contents = f.read()

        # delete the file
        os.unlink(destination)

        return contents
    else:
        return None


def bootstrap(dst: pathlib.Path):
    windows_fallback: bool = False  # use powershell download if curl fails
    models = dst / "models"
    bins = dst / "bin"

    # create the directories
    models.mkdir(parents=True, exist_ok=True)
    bins.mkdir(parents=True, exist_ok=True)

    # Bootstrap CURL
    # on windows versions lower than 10 this is absolutely necessary because powershell downloads are slow as shit
    # windows 10 and later comes with curl
    # HOWEVER i admire the features of the latest curl so i can't resist
    curl_path = (bins / CURL_FILENAME).relative_to(dst)
    if os_name == "Windows":  # `and int(os_version.split(".")[2]) < 17063`
        command = f"$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest {CURL_URL} -OutFile {curl_path}"
    else:
        command = f'{shutil.which("curl")} -L "{CURL_URL}" -o "{curl_path}"'
    run_cmd(command)

    if os_name != "Windows":
        # make curl executable
        os.chmod(curl_path, 0o755)
        # https://github.com/jart/cosmopolitan?tab=readme-ov-file#linux

    # try using new curl
    try:
        subprocess.check_call(
            [curl_path, "-ZL", '"https://captive.apple.com/hotspot-detect.html"'],
            shell=True,
            executable=SHELL,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        backup_curl = "curl.exe" if os_name == "Windows" else "curl"
        curl_path = shutil.which(backup_curl)
        if curl_path:
            curl_path = pathlib.Path(curl_path)
        else:
            curl_path = None
        if curl_path and os_name == "Windows":
            # try using curl
            try:
                subprocess.check_call(
                    [
                        curl_path,
                        "-ZL",
                        '"https://captive.apple.com/hotspot-detect.html"',
                    ],
                    shell=True,
                    executable=SHELL,
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except subprocess.CalledProcessError:
                windows_fallback = True
        elif not curl_path and os_name == "Windows":
            windows_fallback = True
        elif not curl_path:
            print("Could not prepare CURL for downloading artifacts", file=sys.stderr)
            sys.exit(1)

    if not curl_path:
        print("Failed to get CURL", file=sys.stderr)
        sys.exit(1)

    # download llamafile
    llamafile_url = resolve_github(
        curl_path,
        dst,
        LLAMAFILE_INFO,
        LLAMAFILE_REGEX,
        windows_fallback,
    )
    llamafile_path = (bins / LLAMAFILE_FILENAME).relative_to(dst)
    download(
        curl_path,
        dst,
        llamafile_url,
        llamafile_path,
        windows_fallback=windows_fallback,
    )

    if os_name != "Windows":
        # make llamafile executable
        os.chmod(llamafile_path, 0o755)

    # download the AI Server
    aiserver_url = resolve_github(
        curl_path,
        dst,
        AISERVER_INFO,
        AISERVER_REGEX,
        windows_fallback,
    )
    aiserver_version = get_github_version(curl_path, dst, AISERVER_INFO)
    aiserver_path = (dst / AISERVER_FILENAME).relative_to(dst)
    download(
        curl_path,
        dst,
        aiserver_url,
        aiserver_path,
        windows_fallback=windows_fallback,
    )

    # pin the server version at "./.version"
    version_path = (dst / VERSION_FILENAME).relative_to(dst)
    if version_path.exists():
        version_path.unlink()
    with version_path.open("w") as f:
        f.write(f"{aiserver_version}")

    # download the model
    model_path = (models / MODEL_FILENAME).relative_to(dst)
    download(
        curl_path,
        dst,
        MODEL_URL,
        model_path,
        windows_fallback=windows_fallback,
        resume_flag=True,
    )

    print("RWAI bootstrap successful!")


def main():

    # convert relative to absolute path
    directory = pathlib.Path(os.getcwd()).resolve()

    # bootstrap
    bootstrap(directory)


if __name__ == "__main__":
    main()
    sys.exit(0)
