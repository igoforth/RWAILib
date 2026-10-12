# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

RWAILib runs a local Phi-3 model (via llamafile) and wires it into RimWorld 1.5. It is a superproject of three git submodules, each a separate GitHub repo:

| Dir | Repo | Role |
|-----|------|------|
| `AICore/` | igoforth/RWAI_Core | C# Harmony mod: bootstraps, launches and supervises the Python server, owns the gRPC channel, settings UI |
| `AIItems/` | igoforth/RWAI_Items | C# Harmony mod: patches `CompArt.GenerateImageDescription` to swap in AI text. Depends on AICore |
| `AIServer/` | igoforth/RWAI_Server | Python 3.11 gRPC server + llamafile driver, shipped as a self-contained `AIServer.pyz` |

Commit changes inside the submodule first, then bump the pointer in this superproject. `.gitmodules` uses relative URLs (`../RWAI_Core.git`), so cloning needs `--recurse-submodules`.

A local, untracked `MIGRATION-1.7.md` holds the plan for moving to RimWorld 1.7 (net10.0, Harmony 3, JSON-RPC instead of gRPC, LiteDB instead of the gzip cache). The rest of this file describes the current 1.5 codebase; when that file is present, consult it before changing anything it marks for removal.

`bootstrap.py` at the repo root is published as a release asset and downloaded by the mod at runtime. The two `PLACEHOLDER_STRING_*` tokens near the bottom are substituted by `BootstrapTool.InitializePlaceholders` before it runs.

## Build

Requires the .NET SDK (9.x works locally; CI uses 8.x). No RimWorld install is needed: `Krafs.Rimworld.Ref` supplies reference assemblies.

```sh
dotnet build -f net472 -c Release            # whole solution, AICore then AIItems
dotnet build AICore/Source/1.5.csproj -f net472 -c Release
dotnet build -f net472 -c Debug              # defines DEBUG: verbose LogTool output, server runs with --loglevel DEBUG
dotnet cake                                   # same two builds via build.cake; needs `dotnet tool restore` first
```

`.config/` and `tools/` are gitignored, so a fresh clone must recreate the Cake tool manifest (`dotnet new tool-manifest && dotnet tool install cake.tool`) before `dotnet cake` works.

There are no tests.

### What a build produces

Each mod's `Source/Common.props` runs a post-build pipeline: ILRepack merges every dependency DLL into one `<Mod>/1.5/Assemblies/<Mod>.dll`, `About.xml` gets `modVersion` from `Directory.Build.props`, everything is copied to `<Mod>/Build/`, zipped to `<Mod>/<Mod>.zip`, and then `Build/` and `1.5/` are deleted. The zip is the installable artifact; unzip it into RimWorld's `Mods/` to test in-game. AICore's build also extracts the gRPC native libraries into `AICore/Libraries/*.gz`, which `BootstrapTool.SetGrpcOverrideLocation` decompresses at runtime.

### Compiler strictness

`TreatWarningsAsErrors`, `AnalysisMode=AllEnabledByDefault`, and `WarningsAsErrors=nullable` are all on, so any analyzer warning fails the build. The root `.editorconfig` tunes severities: `var` is required for locals, namespaces are file-scoped, `CA2007` (ConfigureAwait) is a warning. Use `#pragma warning disable` with the rule ID for deliberate exceptions, as the existing code does.

`EnableDefaultCompileItems` is false. New `.cs` files must be added to the `<Compile Include=...>` list in `Source/Common.props` or they will not compile. `Source/unused/` holds parked code that is intentionally not in that list.

### AIServer (Python)

Python is pinned to `==3.11.*` (`.python-version`) and managed with uv: `uv sync` creates the venv, `uv run <cmd>` runs tools inside it, `uv lock` refreshes `uv.lock`. The build backend is `uv_build`. `AIServer/README.md` has the full packaging recipe: `uv pip install --target build .` to stage the package and its runtime deps, byte-compile, swap in `loader.py` as `__main__.py`, compile translations, vendor per-platform CPython and grpcio from `AIServer/vendor/`, re-zip. The server must run from inside the pyz: `client.py` reads its GBNF grammars with `ZipFile(sys.argv[0])`, and it expects the cwd to contain `bin/llamafile` and `models/`.

### Protobuf

The C#/Python contract is `job.proto`, kept as two identical copies: `AICore/Protos/v3/job.proto` (C# stubs generated at build by Grpc.Tools) and `AIServer/protos/job.proto` (Python stubs are checked in at `AIServer/src/AIServer/job/job_pb2*.py` and must be regenerated with grpcio-tools). Change both copies together. New job types go in the `oneof job_payload` / `oneof job_result` pairs; the server dispatches on `request.WhichOneof("job_payload")` in `client.py` `AIClient.start`.

### Translations

27 languages are machine-translated from English with an OpenAI key in `OPENAI_API_KEY`:
- C# mods: run `python Tools/translate.py` from inside `AICore/` or `AIItems/` (it resolves `Languages/` relative to cwd) to regenerate `Languages/*/Keyed/*.xml` from `Languages/English`.
- AIServer: pybabel extract/update, `tools/openai_translate.py`, `tools/compile_translations.py`, `tools/convert_translations.py` (packs `.mo` files into a page-aligned blob read by `templates.py`).

## Runtime architecture

### Where things live on disk

AICore creates `RWAI/` next to RimWorld's config folder (`Directory.GetParent(GenFilePaths.ConfigFolderPath)/RWAI`, e.g. `~/.config/RimWorld/RWAI/`) holding `bin/{python,curl,llamafile}`, `models/`, `AIServer.pyz`, `.version`, and the downloaded `bootstrap.py`. `BootstrapTool.CheckConfigured` treats the install as valid only when all of these exist and a model file matching the selected size is present and large enough. AIItems caches generated text in `RWAI Items/items.dat` (`CachedGZipStorage`, keyed by CityHash of `ThingID`).

### AICore lifecycle

Three static managers each expose `UpdateRunningState(bool)`: `BootstrapTool` (download/update), `ServerManager` (Python process), `JobClient` (gRPC channel to `127.0.0.1:50051`). `AICoreMod`'s constructor calls `BootstrapTool.Init()` before `GetSettings` (hardware detection and native lib extraction must precede settings), applies Harmony patches, then in `LongEventHandler.ExecuteWhenFinished` either starts an update check (which starts server and client when done) or starts server and client directly. `AICoreSettings.Enabled`'s setter is the user-facing switch that drives the same managers. `Settings.IsConfigured` is `Enabled && BootstrapTool.IsConfigured`.

Update path: `BootstrapTool` compares `.version` against the latest GitHub release, downloads `bootstrap.py` and a cosmopolitan `python` binary, substitutes placeholders, and runs the script, which fetches curl, llamafile, the server pyz, and a model (HuggingFace, or ModelScope for Chinese-language users).

`ServerManager` launches `bin/python AIServer.pyz --loglevel ... --modelsize ...` through `sh` or `powershell.exe` with cwd `RWAI/`. Server stderr is forwarded to the RimWorld log. Shutdown sends SIGINT via `ProcessInterruptHelper`, waits 1s, then force-kills.

Periodic work runs from a Harmony postfix on `UIRoot.UIRootUpdate` (`GenerallTimeUpdates_Patch` in `Patches.cs`) using `UpdateTaskTime` entries: server status every 2s, `JobClient.MonitorJobsAsync` every 5s, language refresh every 5s. `Main.Perform` marshals work onto the Unity main thread through a coroutine-drained queue. `Tools.SafeAsync` is the fire-and-forget wrapper for async work.

### AIServer internals

`loader.py` becomes the pyz's `__main__` and re-execs the vendored platform-specific CPython bundled inside the zip. `AIServer.main` then runs three supervised asyncio tasks joined by two queues: `run_llama_forever` (llamafile HTTP server on `:50052`), `server.run` (gRPC `JobManager` on `:50051`, puts requests on the input queue and waits for the matching job id on the output queue), and `client.run` (pulls jobs, calls llamafile's OpenAI-compatible API with GBNF grammars from `schemas/` and Babel-localized prompts from `templates.py`, pushes responses). `supervisor` restarts a task on failure and exits the process if it crash-loops.

Model selection: `--modelsize` picks the largest (or for SMALL, smallest) file under `models/` whose name contains the size; the chat template in `LLAMAFILE_TEMPLATES` is chosen from that name.

### AIItems job flow

`CompArt_GenerateImageDescription_Patch` only acts when `UpdateServerStatus.serverStatusEnum == Online`. It hashes the thing, checks `UpdateItemDescriptions` status (Done → overwrite title/description and `ITab_Art` cache; NotDone → serialize the thing with `Scribe.saver.DebugOutputFor`, generate the vanilla title/description, submit via `AICoreMod.Client.AddJob` with `Finish` as callback; Working → skip).

### Shared code between the two mods

`Tools.cs`, `UX.cs`, and `LogTool.cs` in AIItems are copies of the AICore files with only the namespace and mod-class references changed. AIItems references AICore at compile time only (`ExcludeAssets=runtime`), so it does not merge AICore into its DLL. Edits to one copy should be mirrored in the other.
