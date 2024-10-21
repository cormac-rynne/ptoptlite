from setuptools import find_packages, setup

setup(
    name="ptoptlite",
    version="0.1.0",
    author="Cormac Rynne",
    author_email="cormac.ry@gmail.com",
    description="A package for optimisation tasks with PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cormac-rynne/ptoptlite",
    packages=find_packages(include=["ptoptlite", "ptoptlite.*"]),
    install_requires=[
        "torch",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
