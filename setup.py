#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

# Read the version from __init__.py
with open(os.path.join('personal_chatter', '__init__.py'), 'r') as f:
    version_file = f.read()
    version_match = re.search(r'__version__ = [\'"]([^\'"]*)[\'"]', version_file, re.MULTILINE)
    
    if version_match:
        version = version_match.group(1)
    else:
        version = '0.1.0'  # Default if not found

# Read the content of readme.md
with open('readme.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

# Read requirements.txt
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [line for line in requirements if line and not line.startswith('#')]

setup(
    name='personal-chatter',
    version=version,
    author='Andy O',
    author_email='personal-chatter@example.com',
    description='AI chat application with local model support and API interfaces',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/personal-chatter',
    packages=find_packages(exclude=['tests', '*.tests', '*.tests.*', 'tests.*', 'docs', 'examples']),
    include_package_data=True,
    install_requires=requirements,
    python_requires='>=3.12',
    entry_points={
        'console_scripts': [
            'personal-chatter=main:main',
            'pc-chat=main:main',
            'pc-api=api.api:start_backend_api_cli',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Communications :: Chat',
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "mypy",
            "ruff",
            "pre-commit",
        ],
        "test": [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "pytest-benchmark",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
        ],
    },
)
