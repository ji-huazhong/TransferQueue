import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "transfer_queue/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "pyzmq",
    "ray",
    "torch",
    "hydra-core",
    "tensordict>=0.8.0,<=0.9.1,!=0.9.0",
    "numpy<2.0.0",
]


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="transfer_queue",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    author="The TransferQueue team",
    license="Apache 2.0",
    description="Transfer Queue",
    install_requires=install_requires,
    package_data={
        "": ["version/*"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
