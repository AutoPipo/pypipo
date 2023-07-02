# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, 'pypipo', '__version__.py'), 'r') as f:
    exec(f.read(), about)

with open('README.md', 'r', encoding='utf8') as f:
    readme = f.read()

requires = [
    'numpy >= 1.24.3',
    'opencv-python >= 4.7.0',
    'multiprocessing-generator >= 0.3',
    'tqdm >= 4.65.0',
    'scipy >= 1.10.1',
    'click >= 8.1.3',
]

def setup_package():
    metadata = dict(name=about['__title__'],
                    version=about['__version__'],
                    description=about['__description__'],
                    long_description=readme,
                    long_description_content_type="text/markdown",
                    url=about['__url__'],
                    author=about['__author__'],
                    author_email=about['__author_email__'],
                    license=about['__license__'],
                    packages=find_packages(exclude=('sample',)),
                    install_requires=requires,
                    entry_points={
                        'console_scripts': [
                            'camelot = camelot.cli:cli',
                        ],
                    },
                    classifiers = [
                        'Programming Language :: Python :: 3.6',
                        'Programming Language :: Python :: 3.7',],
                    )

    try:
        from setuptools import setup
    except ImportError:
        from distutils.core import setup

    setup(**metadata)

if __name__ == '__main__':
    setup_package()