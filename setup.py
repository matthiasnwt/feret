from setuptools import setup, find_packages

import pathlib

here = pathlib.Path(__file__).parent.resolve()
# The text of the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# This call to setup() does all the work
setup(
    name="feret",
    version="0.1.1",
    description="Calculates the maximum and minimum feret diameter",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/matthiasnwt/feret",
    author="Matthias Nwt",
    license="MIT",
    keywords="feret, feretdiameter, maxferet, minferet",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib"],
    project_urls={
        "Bug Reports": "https://github.com/matthiasnwt/feret/issues",
        "Source": "https://github.com/matthiasnwt/feret/",
    },
)