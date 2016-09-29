#!/usr/bin/env python


from distutils.core import setup, Extension


if __name__ == "__main__":
    ckde = Extension(
        name='kde.kde',
        sources=['kde/kde.c'],
        extra_compile_args=['-Wall', '-O3', '-fPIC']
    )

    setup(
        name='kde',
        version='0.1',
        description='Advanced accelerated Kernel Density Estimation (KDE)',
        author='Sebastian Schoenen',
        author_email='schoenen@physik.rwth-aachen.de',
        url='http://code.icecube.wisc.edu/svn/sandbox/schoenen/kde',
        install_requires=[
            'numexpr',
            'numpy',
            'scipy',
        ],
        extras_require={'cuda': ['pycuda']},
        ext_modules=[ckde],
        packages=['kde'],
        scripts=['kde/test_kde.py']
    )
