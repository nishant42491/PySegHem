from setuptools import find_packages, setup
setup(
    name='PySegHem',
    packages=find_packages(include=['PySegHem']),
    version='0.1.0',
    description='Good Library',
    author='Me',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
)