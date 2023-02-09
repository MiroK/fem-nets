#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name = 'fem-nets',
      version = '0.1',
      description = 'FEniCS FEM space in PyTorch',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/fem-nets.git',
      packages=find_packages(),
      include_package_data=True
)
