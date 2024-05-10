import asyncio
import re

import aiohttp

pkg_s = r"^(?P<package_name>[\w\-\.]+)(?P<subset>\[\w+\])?[ (]*(?:!=[\"']?(?P<not_equals_version>(?:[\w\*]+\.?)+),?[\"']?)?(?:<(?P<max_equal>=)?[\"']?(?P<max_version>(?:[\w\*]+\.?)+[\"']?),?[\"']?)?(?:>(?P<min_equal>=)?[\"']?(?P<min_version>(?:[\w\*]+\.?)+[\"']?)[\"']?)?(?:==[\"']?(?P<equals_version>(?:[\w\*]+\.?)+)[\"']?)?[ )]*(?: ?; ?)?(?:\(?platform_python_implementation [!=><]+ [\"'](?P<platform_python_implementation>[\w\.]+)[\"']\)?(?: and )?)?(?:\(?platform_system [!=><]+= [\"'](?P<platform_system>\w+)[\"']\)?(?: and )?)?(?:\(?python_version [!=><]+ [\"'](?P<python_version>[\w\.]+)[\"']\)?(?: and )?)?(?:\(?extra [!=><]+ [\"'](?P<extra>\w+)[\"']\)?)?$"
pkg_r = re.compile(pkg_s)

async def is_pure_python(package_name, session, visited=None):
    if visited is None:
        visited = set()
    if package_name in visited:
        return True
    visited.add(package_name)

    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to fetch data for {package_name}. Status code: {response.status}")
                return False
            data = await response.json()
    except Exception as e:
        print(f"Error fetching {package_name}: {e}")
        return False

    releases = data.get('releases', {})
    if not releases:
        print(f"No releases found for {package_name}.")
        return False

    for release_version, files in reversed(list(releases.items())):
        for file in files:
            if file['packagetype'] == 'bdist_wheel' and 'none-any' in file['filename']:
                info = data.get('info', {})
                requires_dist = info.get('requires_dist', []) or []
                if not requires_dist:
                    return True
                for req in requires_dist:
                    if 'extra' in req:
                        continue
                    if 'platform_system' in req:
                        print(f"Dependency {req} of {package_name} version {release_version} is not pure Python")
                        return False
                    match = pkg_r.match(req)
                    if not match:
                        print(f"{req}")
                        continue
                    dep_name = match.group('package_name')
                    if not await is_pure_python(dep_name, session, visited):
                        print(f"Dependency {req} of {package_name} version {release_version} is not pure Python")
                        return False
                return True

    print(f"No 'none-any' wheels found for {package_name}.")
    return False

async def main():
    package = 'OpenAI-Python-Client'
    async with aiohttp.ClientSession() as session:
        if await is_pure_python(package, session):
            print(f"{package} is pure Python")
        else:
            print(f"{package} is not pure Python")

# Run the main function
asyncio.run(main())
