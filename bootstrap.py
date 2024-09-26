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
        file="curl.exe" if os_name == "Windows" else "curl",
        name=re.compile(r"curl\.exe" if os_name == "Windows" else r"curl"),
        size=6300000,
        url="https://cosmo.zip/pub/cosmos/v/3.9.2/bin/curl",
    )
    """6.3 MB"""

    LLAMAFILE = File(
        file="llamafile.exe" if os_name == "Windows" else "llamafile",
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
    command: str | list[str],
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
            stdout = result.stdout.strip()
            status = True if result.returncode == 0 else False
            if return_output:
                return (status, stdout)
            return status

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


def ensure_permissions(exe_path: str | pathlib.Path):
    import os

    if os_name == "Windows":
        # remove Mark of the Web, which causes unreliable execution
        command = f"Unblock-File -Path '{exe_path}'"
        run_cmd(command)

        # Set integrity level to Medium
        command = f"icacls {exe_path} /setintegritylevel Medium"
        run_cmd(command)
    else:
        # make curl executable
        os.chmod(exe_path, 0o755)


def bootstrap(dst: pathlib.Path, model: Model, use_chinese_domains: bool):
    import shutil
    import subprocess
    import sys

    windows_fallback: bool = False  # use bitsadmin if curl fails
    models = dst / "models"
    bins = dst / "bin"

    # create the directories
    if not models.exists():
        models.mkdir(parents=True, exist_ok=True)
    bins.mkdir(parents=True, exist_ok=True)

    def test_curl(curl: str | pathlib.Path):
        if not (isinstance(curl, pathlib.Path) and curl.is_relative_to(dst)):
            curl_path = platform_path(pathlib.Path(curl))
        else:
            curl_path = curl

        try:
            subprocess.check_call(
                [
                    curl_path,
                    "-L",
                    "--ssl-revoke-best-effort",
                    "https://captive.apple.com/hotspot-detect.html",
                ],
                shell=True,
                executable=SHELL,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except subprocess.CalledProcessError:
            return False

    # Bootstrap CURL
    # windows 10 and later comes with curl
    curl_path = shutil.which("curl.exe" if os_name == "Windows" else "curl")

    if not (curl_path and test_curl(curl_path)):
        # Get cosmopolitan curl
        curl_path = (bins / Files.CURL.value.file).relative_to(dst)

        # on windows versions lower than 10 this is very helpful because powershell downloads are slow as shit
        if os_name == "Windows":  # `and int(os_version.split(".")[2]) < 17063`
            command = f"$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest {Files.CURL.value.url} -OutFile {curl_path}"
            # command = (
            #     f'certutil -urlcache -split -f "{Files.CURL.value.url}" "{curl_path}"'
            # )
        else:
            command = (
                f'{shutil.which("curl")} -L "{Files.CURL.value.url}" -o "{curl_path}"'
            )

        run_cmd(command)
        ensure_permissions(curl_path)

        if not test_curl(curl_path):
            if os_name == "Windows":
                # Fallback to Windows-specific methods (e.g., bitsadmin)
                windows_fallback = True
            else:
                print(ErrMsg.CURL_PREPARATION_FAILED.value, file=sys.stderr)
                sys.exit(3)
    else:
        # Use existing curl
        curl_path = platform_path(pathlib.Path(curl_path))

    if not curl_path.exists():
        print(ErrMsg.CURL_RETRIEVAL_FAILED.value, file=sys.stderr)
        sys.exit(4)

    progress_reporter = ProgressReporter(
        items=[
            Files.LLAMAFILE.value.size,
            Models.TEST.value.real_size,
            Files.AISERVER.value.size,
            model.real_size,
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

    ensure_permissions(llamafile_path)

    # model related information
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

    model: Model

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
    parser.add_argument(
        "--modelsize",
        choices=["CUSTOM", "MINI", "SMALL", "MEDIUM"],
        default=None,
        help="The model size chosen by heuristics in the RimWorldAI Core mod",
        type=str,
        required=False,
    )
    args = parser.parse_args()

    if args.language is None:
        args.language = "PLACEHOLDER_STRING_LANGUAGE"
    if args.modelsize is None:
        args.modelsize = "PLACEHOLDER_STRING_MODELSIZE"

    # convert relative to absolute path
    directory = pathlib.Path(os.getcwd()).resolve()

    # determine region
    use_chinese_domains: bool = True if "Chinese" in args.language else False

    # determine model size
    model = (
        Models.MINI_CHINA.value
        if use_chinese_domains == True
        else (
            Models.MEDIUM.value
            if args.modelsize == "MEDIUM"
            else Models.SMALL.value if args.modelsize == "SMALL" else Models.MINI.value
        )
    )

    # bootstrap
    bootstrap(directory, model, use_chinese_domains)


if __name__ == "__main__":
    import sys

    main()
    sys.exit(0)
