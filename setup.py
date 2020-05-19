from setuptools import setup, find_namespace_packages
from os import popen

raw_version = open('VERSION').read().strip(' \n')
suffix = popen('git rev-parse --short HEAD').read().strip(' \n')

setup(
    name="waterch-tasker",
    version=f"{raw_version}.{suffix}",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine

    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.rst", "*.md"],
        # And include any *.msg files found in the "hello" package, too:
        # "hello": ["*.msg"],
    },
    # metadata to display on PyPI
    author="Chen Runze",
    author_email="chenrz925@icloud.com",
    description="A scalable and extendable experiment task scheduler framework.",
    long_description=open("README.md").read(),
    url="https://github.com/chenrz925/waterch-tasker",  # project home page, if any
    project_urls={
        "Documentation": "https://waterch-tasker.readthedocs.io/zh_CN/latest/",
        "Source Code": "https://github.com/chenrz925/waterch-tasker",
    },
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "waterch-tasker = waterch.tasker:launch",
        ],
    },
    # could also include long_description, download_url, etc.
    install_requires=open("requirements.txt").readlines(),
)
