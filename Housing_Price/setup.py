import os

from setuptools import find_packages, setup

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
setup(name="src", package_dir={"": "src"}, packages=find_packages(where=str(SRC)))
