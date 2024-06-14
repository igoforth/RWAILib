# RWAILib

### Build

Make sure you have the latest dotnet sdk: https://dotnet.microsoft.com/en-us/download

Probably shouldn't attempt to build it all at once. Do something like the below.

```sh
git clone --recurse-submodules https://github.com/igoforth/RWAILib.git
cd RWAILib

# Build AICore
cd AICore
dotnet restore ./Source/1.5.csproj
dotnet build ./Source/1.5.csproj -f net472 -c Release
dotnet clean ./Source/1.5.csproj
mv AICore.zip ..
cd ..

# AIItems build will also build AICore
cd AIItems
dotnet restore ./Source/1.5.csproj
dotnet build ./Source/1.5.csproj -f net472 -c Release
dotnet clean ./Source/1.5.csproj
rm ../AICore/AICore.zip
mv AIItems.zip ..
cd ..
```
