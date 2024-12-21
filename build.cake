var target = Argument("target", "Default");
var configuration = Argument("configuration", "Release");
var framework = Argument("framework", "net472");

var solution = File("./RWAILib.sln");
var modVersion = EnvironmentVariable("ModVersion") ?? "1.0.0";
var repository = EnvironmentVariable("Repository") ?? "";

// Project paths
var aiCoreProject = File("./AICore/Source/1.5.csproj");
var aiItemsProject = File("./AIItems/Source/1.5.csproj");

Task("Clean")
    .Does(() => {
        CleanDirectories("./**/bin");
        CleanDirectories("./**/obj");
    });

Task("Restore")
    .Does(() => {
        DotNetRestore(solution.Path.FullPath);
    });

Task("Build")
    .IsDependentOn("Clean")
    .IsDependentOn("Restore")
    .Does(() => {
        var buildSettings = new DotNetBuildSettings {
            Configuration = configuration,
            Framework = framework,
            NoRestore = true
        };

        // Build AICore first
        DotNetBuild(aiCoreProject.Path.FullPath, buildSettings);

        // Then build AIItems
        DotNetBuild(aiItemsProject.Path.FullPath, buildSettings);
    });

Task("Default")
    .IsDependentOn("Build");

RunTarget(target);
