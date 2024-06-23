import enum
import pathlib
import platform
import re
import typing

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
PROGRESS_REGEX = re.compile(rb"(\d+)\.\d+%")


class File(typing.NamedTuple):
    import re

    file: str
    name: re.Pattern[str]
    size: int
    url: str = ""
    repo: str = ""


class Files(enum.Enum):
    """Files we will write and/or download to bootstrap."""

    import re

    CURL = File(
        file="curl.com" if os_name == "Windows" else "curl",
        name=re.compile(r"curl\.exe" if os_name == "Windows" else r"curl"),
        size=6300000,
        url="https://cosmo.zip/pub/cosmos/bin/curl",
    )
    """6.3 MB"""

    LLAMAFILE = File(
        file="llamafile.com" if os_name == "Windows" else "llamafile",
        name=re.compile(r"llamafile-\d+\.\d+\.\d+"),
        size=34600000,
        repo="Mozilla-Ocho/llamafile",
    )
    """34.6 MB"""

    AISERVER = File(
        file="AIServer.pyz",
        name=re.compile(r"AIServer\.pyz"),
        size=103300000,
        repo="igoforth/RWAILib",
    )
    """103.3 MB"""

    VERSION = File(
        file=".version",
        name=re.compile(r"\.version"),
        size=1000,
    )
    """1 KB"""


class ModelSize(enum.IntEnum):
    TEST = enum.auto()  # 1
    MINI = enum.auto()
    SMALL = enum.auto()
    MEDIUM = enum.auto()


class Model(typing.NamedTuple):
    url: str
    file: str
    size: ModelSize
    real_size: int


class Models(enum.Enum):
    """Model options that are picked based on hardware detection heuristics."""

    TEST = Model(
        "https://huggingface.co/TKDKid1000/phi-1_5-GGUF/resolve/main/phi-1_5-Q2_K.gguf",
        "phi-1_5-Q2_K.gguf",
        ModelSize.TEST,
        613000000,
    )
    """613 MB"""

    MINI = Model(
        "https://huggingface.co/bartowski/Phi-3-mini-4k-instruct-v0.3-GGUF/resolve/main/Phi-3-mini-4k-instruct-v0.3-Q4_K_M.gguf",
        "Phi-3-mini-4k-instruct-v0.3-Q4_K_M.gguf",
        ModelSize.MINI,
        2390000000,
    )
    """2.39 GB"""

    MINI_CHINA = Model(
        "https://modelscope.cn/api/v1/models/OllmOne/Phi-3-mini-4k-instruct-gguf/repo?Revision=master&FilePath=Phi-3-mini-4k-instruct-q4.gguf",
        "Phi-3-mini-4k-instruct-q4.gguf",
        ModelSize.MINI,
        2320000000,
    )
    """2.32 GB"""

    SMALL = Model(
        # for now, no GGUF for Phi-3-small-128k yet :(
        "https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF/resolve/main/Phi-3-medium-128k-instruct-IQ2_XS.gguf",
        "Phi-3-medium-128k-instruct-IQ2_XS.gguf",
        ModelSize.SMALL,
        4130000000,
    )
    """4.13 GB"""

    MEDIUM = Model(
        "https://huggingface.co/bartowski/Phi-3-medium-128k-instruct-GGUF/resolve/main/Phi-3-medium-128k-instruct-Q4_K_M.gguf",
        "Phi-3-medium-128k-instruct-Q4_K_M.gguf",
        ModelSize.MEDIUM,
        8570000000,
    )
    """8.57 GB"""


class ErrMsg(enum.Enum):
    CONNECTION_FAILED = "Failed to connect to the server: $server"
    """Error message when connection to the server fails."""

    CURL_PREPARATION_FAILED = "Could not prepare CURL for downloading artifacts"
    """Error message when CURL preparation for downloading artifacts fails."""

    CURL_RETRIEVAL_FAILED = "Failed to get CURL"
    """Error message when retrieving CURL fails."""

    FILE_NOT_FOUND = "File not found: $filename"
    """Error message when a specified file is not found."""

    INVALID_INPUT = "Invalid input provided: $input"
    """Error message when an invalid input is provided."""

    LLAMAFILE_DECODE_FAILED = "Failed to decode llamafile output"
    """Error message when decoding llamafile output fails."""

    LLAMAFILE_EXECUTION_FAILED = "Could not run llamafile"
    """Error message when executing llamafile fails."""

    PARSE_RESPONSE_ERROR = "Failed to parse the response from $api_url due to $error"
    """Error message when parsing the response from a specified API URL fails."""

    PLATFORM_UNSUPPORTED = "Unsupported platform: $platform"
    """Error message when the platform is unsupported."""

    RAM_RETRIEVAL_FAILED_DARWIN = "Failed to retrieve available RAM from Darwin. Returning a conservative estimate"
    """Error message when retrieving available RAM from Darwin fails, returning a conservative estimate."""

    RAM_RETRIEVAL_FAILED_WINDOWS = "Failed to retrieve available RAM from Windows. Returning a conservative estimate"
    """Error message when retrieving available RAM from Windows fails, returning a conservative estimate."""

    RELEASE_NOT_FOUND = "Failed to find a release URL for GitHub repository $repo"
    """Error message when failing to find a GitHub release url for the a repository."""

    RESPONSE_ERROR = "Failed to get a response from $api_url"
    """Error message when failing to get a response from a specified API URL."""

    RUN_COMMAND_FAILED = "Failed to run $command. stderr:\n$stderr"
    """Error message when a command execution fails."""

    UNKNOWN_ERROR = "An unknown error occurred. Traceback:\n$traceback"
    """Error message when an unknown error occurs."""

    def format(self, **kwargs: str):
        import string

        return string.Template(self.value).safe_substitute(**kwargs)


class ProgressReporter:
    def __init__(self, items: list[int]):
        self.items: list[int] = items
        self.total_size: int = sum(size for size in self.items)
        self.downloaded_sizes: list[int] = [0] * len(self.items)
        self.current_total_downloaded: int = 0
        self.last_reported_percentage: int = 0
        self.size_history: list[int] = []

    def update_total_size(self):
        self.total_size = sum(size for size in self.items)

    def add_item(self, item: int):
        """Add a new item to the tracking list."""
        self.items.append(item)
        self.downloaded_sizes.append(0)  # Initialize downloaded size for new item
        self.update_total_size()

    def update_item_size(self, index: int, new_size: int):
        """Update the size of an existing item."""
        if index < len(self.items):
            self.items[index] = new_size
            self.update_total_size()

    def update_progress(self, index: int, size: int):
        if index >= len(self.items):
            return  # Avoid index errors

        if size <= self.items[index]:
            change_in_size = size - self.downloaded_sizes[index]
            self.downloaded_sizes[index] = size
        else:
            max_size = self.items[index]
            change_in_size = max_size - self.downloaded_sizes[index]
            self.downloaded_sizes[index] = max_size

        self.current_total_downloaded += change_in_size
        self.size_history.append(self.current_total_downloaded)

        if self.should_update():
            percent_complete = int(
                (self.current_total_downloaded / self.total_size) * 100
            )
            if percent_complete != self.last_reported_percentage:
                self.last_reported_percentage = percent_complete
                print(percent_complete)

        if len(self.size_history) > 3:
            self.size_history.pop(0)

    def should_update(self):
        if len(self.size_history) < 3:
            return True
        return (
            self.size_history[-1] > self.size_history[-3]
            and self.size_history[-2] > self.size_history[-3]
        )


def platform_path(posix_path: pathlib.Path):
    if os_name == "Windows":
        import pathlib

        # Convert the PosixPath to a string
        path_str = str(posix_path)

        # Handle relative paths by checking if the path starts with '/'
        if path_str.startswith("/"):
            parts = path_str.lstrip("/").split("/", 1)
            if (
                len(parts[0]) == 1 and parts[0].isalpha()
            ):  # Check if the first part is a single letter (drive letter)
                drive_letter = parts[0].upper() + ":"
                rest_of_path = parts[1] if len(parts) > 1 else ""
            else:
                drive_letter = ""
                rest_of_path = path_str
        else:
            drive_letter = ""
            rest_of_path = path_str

        # Replace forward slashes with backslashes
        windows_path_str = rest_of_path.replace("/", "\\")

        # Combine the drive letter and the rest of the path
        if drive_letter:
            windows_path = pathlib.Path(drive_letter + "\\" + windows_path_str)
        else:
            windows_path = pathlib.Path(windows_path_str)

        return windows_path
    return posix_path


def get_github_version(curl_path: pathlib.Path, cwd: pathlib.Path, repo: str) -> str:
    import json
    import sys

    api_url: str = f"https://api.github.com/repos/{repo}/releases/latest"
    user_agent: str = "igoforth/RWAILib"

    response = download(curl_path, cwd, api_url, user_agent=user_agent)

    if response is None:
        print(
            ErrMsg.RESPONSE_ERROR.format(api_url=api_url),
            file=sys.stderr,
        )
        sys.exit(14)
    else:
        response = response.strip()
    if response == "":
        print(
            ErrMsg.RESPONSE_ERROR.format(api_url=api_url),
            file=sys.stderr,
        )
        sys.exit(14)

    try:
        return json.loads(response)["tag_name"]
    except json.JSONDecodeError as e:
        print(
            ErrMsg.PARSE_RESPONSE_ERROR.format(api_url=api_url, error=e.msg),
            file=sys.stderr,
        )
        sys.exit(9)


def get_total_ram() -> int:
    """Returns the total amount of RAM in bytes."""
    import sys

    if os_name == "Windows":
        command = "wmic ComputerSystem get TotalPhysicalMemory"
        output = run_cmd(command, return_output=True)
        if type(output) is not bool and output[0] is True:
            return int(output[1].split("\n")[1].strip())
        else:
            print(ErrMsg.RAM_RETRIEVAL_FAILED_WINDOWS.value, file=sys.stderr)
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
        if type(output) is not bool and output[0] is True:
            return int(output[1].split()[1])
        else:
            print(ErrMsg.RAM_RETRIEVAL_FAILED_DARWIN.value, file=sys.stderr)
            return 4000000000

    else:
        raise NotImplementedError(ErrMsg.PLATFORM_UNSUPPORTED.format(os_name=os_name))


# this is going to be the worst heuristic I've ever made
def estimate_vram(gpu_model: str) -> ModelSize | None:
    import re

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
        "95": ModelSize.MEDIUM,  # AMD uses 95 too apparently
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
    progress_reporter: ProgressReporter | None = None,
) -> Model:
    import subprocess
    import sys

    models: pathlib.Path = dst / "models"
    test_model_path: pathlib.Path = (models / Models.TEST.value.file).relative_to(dst)
    llamafile_test_params_list: list[str] = [
        "-ngl",
        "9999",
        "-m",
        str(test_model_path),
        "--cli",
    ]

    def finish(mdl: ModelSize) -> Model:
        if test_model_path.exists():
            test_model_path.unlink()
        return next(model for model in Models if model.value.size == mdl).value

    # success:
    # "Apple Metal GPU support successfully loaded"
    # "welcome to CUDA SDK with tinyBLAS"
    # failure:
    # "tinyBLAS not supported"

    if progress_reporter:
        download(
            curl_path,
            dst,
            Models.TEST.value.url,
            test_model_path,
            windows_fallback=windows_fallback,
            resume_flag=True,
            progress_reporter=progress_reporter,
            index=1,
            file_size=progress_reporter.items[1],
        )
    else:
        download(
            curl_path,
            dst,
            Models.TEST.value.url,
            test_model_path,
            windows_fallback=windows_fallback,
            resume_flag=True,
        )

    try:
        output: str = subprocess.check_output(
            [llamafile_path, *llamafile_test_params_list],
            stderr=subprocess.STDOUT,
        ).decode("utf-8")
        if output != "":
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
        else:
            raise Exception

    except subprocess.CalledProcessError:
        print(ErrMsg.LLAMAFILE_EXECUTION_FAILED.value, file=sys.stderr)
        sys.exit(8)
    except UnicodeDecodeError:
        print(ErrMsg.LLAMAFILE_DECODE_FAILED.value, file=sys.stderr)
        sys.exit(7)


def resolve_github(
    curl_path: pathlib.Path,
    cwd: pathlib.Path,
    repo: str,
    file_r: re.Pattern[str],
    windows_fallback: bool,
) -> str:
    import json
    import sys

    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    user_agent: str = "igoforth/RWAILib"
    download_url = ""

    # get the latest release from the API
    response = download(
        curl_path,
        cwd,
        api_url,
        windows_fallback=windows_fallback,
        user_agent=user_agent,
    )

    if response is None:
        print(
            ErrMsg.RESPONSE_ERROR.format(api_url=api_url),
            file=sys.stderr,
        )
        sys.exit(14)
    else:
        response = response.strip()
    if response == "":
        print(
            ErrMsg.RESPONSE_ERROR.format(api_url=api_url),
            file=sys.stderr,
        )
        sys.exit(14)

    # parse the response
    try:
        response_json = json.loads(response)

        # get the first asset that matches the pattern
        for asset in response_json["assets"]:
            if file_r.match(asset["name"]):
                download_url = asset["browser_download_url"]
                break

        if not download_url:
            print(ErrMsg.RELEASE_NOT_FOUND.format(repo=repo), file=sys.stderr)
            sys.exit(13)

    except Exception as e:
        print(
            ErrMsg.UNKNOWN_ERROR.format(traceback=str(e)),
            file=sys.stderr,
        )
        sys.exit(16)

    return download_url


def parse_curl_progress(line: bytes):
    # Match the progress output of curl
    match = PROGRESS_REGEX.search(line)
    if match:
        decoded_match = int(match.group(1).decode("utf-8"))
        if decoded_match == 100:
            return 0
        else:
            return decoded_match
    return None


def run_cmd(
    command: str,
    return_output: bool = False,
    progress_reporter: ProgressReporter | None = None,
    index: int | None = None,
    file_size: int | None = None,
) -> tuple[bool, str] | bool:
    import subprocess

    # Detect if the command is using curl and specifically a download command
    # print(f"`{command}`")
    if (
        "curl" in command
        and progress_reporter is not None
        and file_size is not None
        and index is not None
    ):
        try:
            # Execute the curl command and stream the output
            with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                executable=SHELL,
            ) as process:
                partial_line = b""  # Buffer to store incomplete lines
                while True:
                    chunk = process.stderr.read(512)  # type: ignore
                    if not chunk:
                        break
                    partial_line += chunk
                    if b"\r" in partial_line:
                        last_cr = partial_line.rfind(b"\r")
                        line, partial_line = (
                            partial_line[:last_cr],
                            partial_line[last_cr + 1 :],
                        )
                        check = parse_curl_progress(line)
                        if check is not None:
                            incremental_size = int((check / 100.0) * file_size)
                            if incremental_size > 0:
                                progress_reporter.update_progress(
                                    index, incremental_size
                                )

                process.wait()  # Wait for the process to complete

                if process.returncode == 0:
                    progress_reporter.update_progress(0, file_size)
                    return True
                else:
                    error_msg = process.stderr.read().decode("utf-8")  # type: ignore
                    print(f"Failed to download: {error_msg}", file=sys.stderr)
                    return False
        except subprocess.CalledProcessError as e:
            print(f"Failed to download: {e}", file=sys.stderr)
            sys.exit(15)
    else:
        try:
            result = subprocess.run(
                command,
                shell=True,
                executable=SHELL,
                capture_output=True,
                text=True,
                check=True,
            )
            if result.returncode != 0:
                if return_output:
                    return (False, result.stdout.strip())
                return False
            else:
                if return_output:
                    return (True, result.stdout.strip())
                return True
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {command}, Error: {e.stderr}", file=sys.stderr)
            sys.exit(15)


def construct_command(
    curl_path: pathlib.Path,
    url: str,
    destination: str,
    dst: pathlib.Path | None = None,
    resume_flag: bool = False,
    user_agent: str = "",
):
    if resume_flag:
        command = f'{str(curl_path)} -# -C - -L --ssl-revoke-best-effort "{url}" -o "{destination}"'
    else:
        if dst and dst.exists():
            dst.unlink()
        command = f'{str(curl_path)} -# -L --ssl-revoke-best-effort "{url}" -o "{destination}"'
    if user_agent:
        command += f' -A "{user_agent}"'
    return command


def download(
    curl_path: pathlib.Path,
    cwd: pathlib.Path,
    url: str,
    dst: pathlib.Path | None = None,
    windows_fallback: bool = False,
    resume_flag: bool = False,
    user_agent: str = "",
    progress_reporter: ProgressReporter | None = None,
    index: int | None = None,
    file_size: int | None = None,
) -> str | None:
    import tempfile

    destination: pathlib.Path | str

    # if no destination is provided, caller wants the contents of the file
    if not dst:
        destination = str(
            platform_path(pathlib.Path(tempfile.NamedTemporaryFile(delete=False).name))
        )
    else:
        destination = str(dst)

    if not windows_fallback:
        command = construct_command(
            curl_path, url, destination, dst, resume_flag, user_agent
        )
    else:
        destination = pathlib.Path(destination).resolve()
        destination = platform_path(destination)
        command = f'bitsadmin /transfer 1 "{url}" "{destination}"'

    run_cmd(
        command,
        progress_reporter=progress_reporter,
        index=index,
        file_size=file_size,
    )

    if not dst:
        import os

        # return the contents of the file and delete the file
        with open(destination, "r") as f:
            contents = f.read()

        # delete the file
        os.unlink(destination)

        return contents
    else:
        return None


def bootstrap(dst: pathlib.Path, use_chinese_domains: bool):
    import os
    import shutil
    import subprocess
    import sys

    windows_fallback: bool = False  # use bitsadmin if curl fails
    models = dst / "models"
    bins = dst / "bin"

    # create the directories
    if not models.exists():
        models.mkdir(parents=True, exist_ok=True)
        models_dir_size: int | None = None
    else:
        models_dir_size: int | None = sum(
            file.stat().st_size for file in models.rglob("*") if file.is_file()
        )
        if models_dir_size < 2000000000:
            models_dir_size = None  # If directory size less than 2 GB, assume models don't need to be updated
    bins.mkdir(parents=True, exist_ok=True)

    # Bootstrap CURL
    # on windows versions lower than 10 this is absolutely necessary because powershell downloads are slow as shit
    # windows 10 and later comes with curl
    # HOWEVER i admire the features of the latest curl so i can't resist
    curl_path = (bins / Files.CURL.value.file).relative_to(dst)
    if os_name == "Windows":  # `and int(os_version.split(".")[2]) < 17063`
        command = f"$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest {Files.CURL.value.url} -OutFile {curl_path}"
    else:
        command = f'{shutil.which("curl")} -L "{Files.CURL.value.url}" -o "{curl_path}"'
    if not curl_path.exists():
        run_cmd(command)

    if os_name != "Windows":
        # make curl executable
        os.chmod(curl_path, 0o755)
        # https://github.com/jart/cosmopolitan?tab=readme-ov-file#linux

    # try using new curl
    try:
        subprocess.check_call(
            [
                curl_path,
                "-L",
                "--ssl-revoke-best-effort",  # some AV will do MITM, this prevent curl from dying
                '"https://captive.apple.com/hotspot-detect.html"',
            ],
            shell=True,
            executable=SHELL,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        backup_curl = "curl.exe" if os_name == "Windows" else "curl"
        curl_path = shutil.which(backup_curl)
        if curl_path and os_name == "Windows":
            # use windows curl with Windows paths
            curl_path = platform_path(pathlib.Path(backup_curl))
            try:
                subprocess.check_call(
                    [
                        curl_path,
                        "-L",
                        "--ssl-revoke-best-effort",  # some AV will do MITM, this prevents curl from dying
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
        elif curl_path:
            curl_path = pathlib.Path(curl_path)
        elif not curl_path and os_name == "Windows":
            # use bitsadmin
            windows_fallback = True
        elif not curl_path:
            # irrecoverable
            print(ErrMsg.CURL_PREPARATION_FAILED.value, file=sys.stderr)
            sys.exit(3)

    if not curl_path:
        print(ErrMsg.CURL_RETRIEVAL_FAILED.value, file=sys.stderr)
        sys.exit(4)

    progress_reporter = ProgressReporter(
        items=[
            Files.LLAMAFILE.value.size,
            Models.TEST.value.real_size,
            Files.AISERVER.value.size,
            Models.MINI.value.real_size if models_dir_size is None else 0,
        ]
    )

    # download llamafile
    llamafile_url = resolve_github(
        curl_path,
        dst,
        Files.LLAMAFILE.value.repo,
        Files.LLAMAFILE.value.name,
        windows_fallback,
    )
    llamafile_path = (bins / Files.LLAMAFILE.value.file).relative_to(dst)
    download(
        curl_path=curl_path,
        cwd=dst,
        url=llamafile_url,
        dst=llamafile_path,
        windows_fallback=windows_fallback,
        progress_reporter=progress_reporter,
        index=0,
        file_size=progress_reporter.items[0],
    )

    if os_name != "Windows":
        # make llamafile executable
        os.chmod(llamafile_path, 0o755)

    # determine model to download
    if not use_chinese_domains:
        model: Model = get_capabilities(
            curl_path,
            dst,
            llamafile_path,
            windows_fallback,
            progress_reporter,
        )
    else:
        # print(
        #     "感谢您在中国使用RimWorldAI！huggingface.co无法访问，因此我们将使用modelscope.cn。遗憾的是，截至2024年6月19日，只有Phi-3-mini可用。感谢您的耐心等待，我们正在等待Phi-3-small和Phi-3-medium的上线。",
        #     file=sys.stderr,
        # )
        model: Model = Models.MINI_CHINA.value
    progress_reporter.update_item_size(2, model.real_size)
    model_path = (models / model.file).relative_to(dst)

    # download the AI Server
    aiserver_url = resolve_github(
        curl_path,
        dst,
        Files.AISERVER.value.repo,
        Files.AISERVER.value.name,
        windows_fallback,
    )
    aiserver_version = get_github_version(
        curl_path,
        dst,
        Files.AISERVER.value.repo,
    )
    aiserver_path = (dst / Files.AISERVER.value.file).relative_to(dst)
    download(
        curl_path,
        dst,
        aiserver_url,
        aiserver_path,
        windows_fallback=windows_fallback,
        progress_reporter=progress_reporter,
        index=2,
        file_size=progress_reporter.items[2],
    )

    # pin the server version at "./.version"
    version_path = (dst / Files.VERSION.value.file).relative_to(dst)
    if version_path.exists():
        version_path.unlink()
    with version_path.open("w") as f:
        f.write(f"{aiserver_version}")

    # download the model
    download(
        curl_path,
        dst,
        model.url,
        model_path,
        windows_fallback=windows_fallback,
        resume_flag=True,
        progress_reporter=progress_reporter,
        index=3,
        file_size=progress_reporter.items[3],
    )

    # print("RWAI bootstrap successful!")


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="A bootstrapper for RimWorldAI")
    parser.add_argument(
        "--language",
        choices=[
            "Arabic",
            "ChineseSimplified",
            "ChineseTraditional",
            "Czech",
            "Danish",
            "Dutch",
            "English",
            "Estonian",
            "Finnish",
            "French",
            "German",
            "Hungarian",
            "Italian",
            "Japanese",
            "Korean",
            "Norwegian",
            "Polish",
            "Portuguese",
            "PortugueseBrazilian",
            "Romanian",
            "Russian",
            "Slovak",
            "Spanish",
            "SpanishLatin",
            "Swedish",
            "Turkish",
            "Ukrainian",
        ],
        default=None,
        help="The language assists in determining the domain to download the model from.",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    if args.language is None:
        args.language = "PLACEHOLDER_STRING_LANGUAGE"

    # convert relative to absolute path
    directory = pathlib.Path(os.getcwd()).resolve()

    # determine region
    use_chinese_domains = True if "Chinese" in args.language else False

    # bootstrap
    bootstrap(directory, use_chinese_domains)


if __name__ == "__main__":
    import sys

    main()
    sys.exit(0)
