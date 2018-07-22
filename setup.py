import sys

from setuptools import find_packages, setup


def main():
    install_requires = [
        'click',
        'regex',
        'treelib',
    ]
    if sys.platform == 'win32':
        install_requires.append('colorama')

    setup_requires = ['setuptools_scm']
    if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
        setup_requires.append('pytest-runner')

    tests_require = ['pytest']

    setup(
        name='fbSAT',
        description='fbSAT',
        url='https://github.com/ctlab/fbSAT',
        author='Konstantin Chukharev',
        author_email='lipen00@gmail.com',
        license='GNU GPLv3',
        python_requires='>=3.6',
        package_dir={'': 'src'},
        packages=find_packages('src'),
        use_scm_version={
            'write_to': 'src/fbsat/version.py',
            'version_scheme': 'post-release',
            # 'local_scheme': lambda _: '',
            'local_scheme': 'dirty-tag',
        },
        install_requires=install_requires,
        setup_requires=setup_requires,
        tests_require=tests_require,
        entry_points={
            'console_scripts': [
                'fbsat = fbsat:cli',
            ]
        },
        zip_safe=False,
    )


if __name__ == '__main__':
    main()
