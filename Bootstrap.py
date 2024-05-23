import enum
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

SHELL = "powershell.exe" if os_name == "Windows" else "sh"

MODEL_TEST_URL = (
    "https://huggingface.co/TKDKid1000/phi-1_5-GGUF/resolve/main/phi-1_5-Q2_K.gguf"
)
MODEL_TEST_FILENAME = "phi-1_5-Q2_K.gguf"
MODEL_MINI_URL = "https://huggingface.co/PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed/resolve/main/Phi-3-mini-128k-instruct.IQ4_NL.gguf"
MODEL_MINI_FILENAME = "Phi-3-mini-128k-instruct.IQ4_NL.gguf"
MODEL_SMALL_URL = "https://huggingface.co/PrunaAI/Phi-3-mini-128k-instruct-GGUF-Imatrix-smashed/resolve/main/Phi-3-mini-128k-instruct.Q8_0.gguf"  # for now, no GGUF for Phi-3-small yet :(
MODEL_SMALL_FILENAME = "Phi-3-mini-128k-instruct.Q8_0.gguf"
MODEL_MEDIUM_URL = "https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF/resolve/main/Phi-3-medium-128k-instruct-IQ4_NL.gguf"
MODEL_MEDIUM_FILENAME = "Phi-3-medium-128k-instruct-IQ4_NL.gguf"

CURL_URL = "https://cosmo.zip/pub/cosmos/bin/curl"
CURL_FILENAME = "curl.com" if os_name == "Windows" else "curl"
LLAMAFILE_INFO = "Mozilla-Ocho/llamafile"
LLAMAFILE_REGEX = r"llamafile-\d+\.\d+\.\d+"
LLAMAFILE_FILENAME = "llamafile.com" if os_name == "Windows" else "llamafile"
LLAMAFILE_TEST_PARAMS_LIST: list[str] = [
    "-ngl",
    "9999",
    "-m",
    "models/Phi-3-mini-128k-instruct.Q4_K_M.gguf",
    "--cli",
]
AISERVER_INFO = "igoforth/RWAILib"
AISERVER_REGEX = r"AIServer\.pyz"
AISERVER_FILENAME = "AIServer.pyz"
VERSION_FILENAME = ".version"


class ModelSize(enum.Enum):
    MINI = 1
    SMALL = 2
    MEDIUM = 3


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


def get_total_ram() -> int:
    """Returns the total amount of RAM in bytes."""

    if os_name == "Windows":
        command = "wmic ComputerSystem get TotalPhysicalMemory"
        output = run_cmd(command, return_output=True)
        if output:
            return int(output.split("\n")[1].strip())
        else:
            print(
                "Failed to retrieve available RAM from Windows. Returning conservative estimate.",
                file=sys.stderr,
            )
            return 4000000000

    elif os_name == "Linux":
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        mem_total_line = next(
            line for line in meminfo.split("\n") if line.startswith("MemTotal")
        )
        mem_total_kb = int(mem_total_line.split()[1])
        return mem_total_kb * 1024

    elif os_name == "Darwin":
        command = "sysctl hw.memsize"
        output = run_cmd(command, return_output=True)
        if output:
            return int(output.split()[1])
        else:
            print(
                "Failed to retrieve available RAM from Darwin. Returning conservative estimate.",
                file=sys.stderr,
            )
            return 4000000000

    else:
        raise NotImplementedError(f"Unsupported platform: {os_name}")


# this is going to be the worst heuristic I've ever made
def estimate_vram(gpu_model: str) -> ModelSize | None:
    gpu_model = gpu_model.strip()
    # gpu_r = re.compile(
    #     r"^.*(?P<family>RX|RTX|GTX) (?P<model>\w+).*compute capability (?P<cc>[\d\.]+).*$",
    #     re.IGNORECASE,
    # )
    gpu_r = re.compile(
        r"^Device 0.*(?P<family>RX|RTX|GTX) (?P<model>\w+).*$",
        re.IGNORECASE,
    )
    mdl_lst = ["50", "60", "70", "80", "90"]
    gpu_model = gpu_model.lower()

    match = gpu_r.match(gpu_model)
    if not match:
        return None

    family = match.group("family").upper()
    model = match.group("model")
    # compute_capability = match.group("cc")

    # Phi-3-medium-128k-instruct-Q4_K_M.gguf 8.56GB
    # Phi-3-small-128k-instruct-Q4_K_M.gguf will be ~4.51GB
    # Phi-3-mini-128k-instruct-Q4_K_M.gguf 2.34GB

    nvidia_vram = {
        "50": ModelSize.MINI,  # "2GB to 4GB",
        "60": ModelSize.MINI,  # "3GB to 6GB",
        "70": ModelSize.SMALL,  # "8GB",
        "80": ModelSize.MEDIUM,  # "8GB to 11GB",
        "90": ModelSize.MEDIUM,  # "24GB",
    }

    amd_vram = {
        "50": ModelSize.MINI,  # "2GB to 4GB",
        "60": ModelSize.MINI,  # "2GB to 6GB",
        "70": ModelSize.SMALL,  # "4GB to 12GB",
        "80": ModelSize.MEDIUM,  # "8GB to 16GB",
        "90": ModelSize.MEDIUM,  # "16GB",
    }

    if family == "GTX" or family == "RTX":
        for mdl in mdl_lst:
            if mdl in model:
                return nvidia_vram.get(mdl, ModelSize.MINI)
    elif family == "RX":
        for mdl in mdl_lst:
            if mdl in model:
                return amd_vram.get(mdl, ModelSize.MINI)

    return None


def get_capabilities(
    curl_path: pathlib.Path,
    dst: pathlib.Path,
    llamafile_path: pathlib.Path,
    windows_fallback: bool,
) -> ModelSize:
    models: pathlib.Path = dst / "models"
    test_model_url: str = (
        "https://huggingface.co/TKDKid1000/phi-1_5-GGUF/resolve/main/phi-1_5-Q2_K.gguf"
    )
    test_model_name: str = "phi-1_5-Q2_K.gguf"
    test_model_path: pathlib.Path = (models / test_model_name).relative_to(dst)
    llamafile_test_params_list: list[str] = [
        "-ngl",
        "9999",
        "-m",
        str(test_model_path),
        "--cli",
    ]

    def finish(mdl: ModelSize) -> ModelSize:
        if test_model_path.exists():
            test_model_path.unlink()
        return mdl

    # success:
    # "Apple Metal GPU support successfully loaded"
    # "welcome to CUDA SDK with tinyBLAS"
    # failure:
    # "tinyBLAS not supported"

    download(
        curl_path,
        dst,
        test_model_url,
        test_model_path,
        windows_fallback=windows_fallback,
        resume_flag=True,
    )

    try:
        output = subprocess.check_output(
            [llamafile_path, *llamafile_test_params_list],
            stderr=subprocess.STDOUT,
        )
        if output is not None:
            # ggml_metal_init: found device: Apple M2 Pro
            # ggml_init_cublas: no CUDA devices found, CUDA will be disabled
            # ggml_init_cublas: found * ROCm devices:
            # Device 0: Radeon RX 7900 XTX, compute capability 11.0, VMM: no
            # ggml_cuda_init: found * CUDA devices:
            # Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
            # Device 0: NVIDIA GeForce GTX 1060, compute capability 6.1
            # 7900 xtx: 24gb vram
            # 4090: 24gb vram
            # 3060: 12gb vram, cc 8.6
            # 1080 Ti: 11gb vram, but same cc as 1060
            # 1060: 6gb vram
            # should make biggest model fit in 6gb vram?

            # ggml_metal_init (Metal)
            # ggml_init_cublas (AMD)
            # ggml_cuda_init (Nvidia)
            # default (CPU)

            output: str = output.decode("utf-8")
            if "ggml_metal_init" in output:
                # just to be performant. optimize later
                return finish(ModelSize.MINI)
            else:
                lines: list[str] = output.split("\n")
                result = None
                for line in lines:
                    result = estimate_vram(line)
                    if result is not None:
                        break
                if result is None:
                    return finish(ModelSize.MINI)
                else:
                    return finish(result)

    except subprocess.CalledProcessError:
        print("Could not run llamafile!", file=sys.stderr)
        sys.exit(1)
    except UnicodeDecodeError:
        print("Could not decode llamafile output!", file=sys.stderr)
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
        resume_flag=True,
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


def run_cmd(command: str, return_output: bool = False) -> str | None:
    string_command = "`" + command + "`"
    try:
        print(string_command)
        result = subprocess.run(
            command,
            shell=True,
            executable=SHELL,
            capture_output=True,
            text=True,
            check=True,
        )
        if return_output:
            return result.stdout.strip()
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
            if dst and dst.exists():
                dst.unlink()
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

    # determine model to download
    model_size: ModelSize = get_capabilities(
        curl_path,
        dst,
        llamafile_path,
        windows_fallback,
    )
    if model_size == ModelSize.MEDIUM:
        model_url = MODEL_MEDIUM_URL
        model_path = (models / MODEL_MEDIUM_FILENAME).relative_to(dst)
    elif model_size == ModelSize.SMALL:
        model_url = MODEL_SMALL_URL
        model_path = (models / MODEL_SMALL_FILENAME).relative_to(dst)
    else:
        model_url = MODEL_MINI_URL
        model_path = (models / MODEL_MINI_FILENAME).relative_to(dst)

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
    download(
        curl_path,
        dst,
        model_url,
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
