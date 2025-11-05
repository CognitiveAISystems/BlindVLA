from setuptools import find_packages, setup

setup(
    name="simpler_env",
    version="0.0.1",
    packages=find_packages(include=["simpler_env*"]),
    python_requires=">=3.10",
)
