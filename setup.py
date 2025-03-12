from setuptools import setup, find_packages

setup(
    name="meetneighbors",          # Replace with your project name
    version="0.1.0",
    package_dir={"": "src"},      # Indicate that packages are inside "src/"
    packages=find_packages(where="src"),  # Find packages only in "src/"
    entry_points = {
        "console_scripts":["meetneighbors=meetneighbors.main:run"]
    },
    install_requires=[],          # Add dependencies here if needed
)
