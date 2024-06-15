# RWAILib

### Build

Make sure you have the latest dotnet sdk: https://dotnet.microsoft.com/en-us/download

Probably shouldn't attempt to build it all at once. Do something like the below.

```sh
git clone --recurse-submodules https://github.com/igoforth/RWAILib.git
cd RWAILib
dotnet build -f net472 -c Release
```
