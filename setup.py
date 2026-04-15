# This file is part of StochasticPBBP.

# StochasticPBBP is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# StochasticPBBP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with SDRP. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

setup(
      name='StochasticPBBP',
      version='0.1',
      author="Yuval Aroosh, Ayal Taitler",
      author_email="yuvalaro@post.bgu.ac.il, ataitler@gmail.com",
      description="Model-based differential policy optimization",
      license="MIT License",
      url="https://github.com/DRIVE-Lab-BGU/StochasticPBBP",
      packages=find_packages(),
      install_requires=['pyrddlgym<3', 'rddlrepository<2.3', 'torch', 'numpy<2.2'],
      python_requires=">=3.9,<3.13",
      package_data={'': ['*.rddl']},
      include_package_data=True,
      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)