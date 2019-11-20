#!/usr/bin/env python


from distutils.core import setup, Extension


if __name__ == "__main__":
    ckde = Extension(
        name='kde.kde',
        sources=['kde/kde.c'],
        extra_compile_args=['-Wall', '-O3', '-fPIC', '-Werror']
    )

    setup(
        name='kde',
        version='0.1',
        description=('Multi-dimensional Kernel Density Estimation (KDE)'
                     ' including adaptive bandwidths and C and'
                     ' CUDA implementations for specific cases.'),
        author='Sebastian Schoenen, Martin Leuermann',
        author_email='schoenen@physik.rwth-aachen.de',
        url='https://github.com/icecubeopensource/kde',
        install_requires=[
            'numexpr',
            'numpy',
            'scipy',
        ],
        extras_require={'cuda': ['pycuda']},
        ext_modules=[ckde],
        packages=['kde'],
        entry_points={
            'console_scripts': ['test_kde.py = kde.test_kde:main']
        }
    )
