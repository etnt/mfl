from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="mfl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.6",
)
