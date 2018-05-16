from setuptools import setup

from version import __version__

setup(
    name='fbSAT',
    version=__version__,
    description='fbSAT',
    url='https://github.com/ctlab/fbSAT',
    author='Konstantin Chukharev',
    author_email='lipen00@gmail.com',
    license='GNU GPLv3',
    python_requires='>=3.6',
    py_modules=['version'],
    packages=['fbsat'],
    install_requires=[
        'regex',
        'click',
        'treelib',
        'colorama',
    ],
    entry_points={
        'console_scripts': [
            'fbsat = fbsat:cli',
        ]
    }
)
