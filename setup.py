from setuptools import find_packages, setup
setup(
    name='PySegHem',
    packages=find_packages(include=['PySegHem']),
    version='0.1.0',
    description='Good Library',
    author='Nishant42491',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    test_suite='tests',
    author_email = 'n.rajadhyaksha@somaiya.edu',      # Type in your E-Mail
    url = 'https://github.com/nishant42491/PySegHem.git',   # Provide either the link to your github or to your website
    download_url = 'https://github.com/nishant42491/PySegHem.git/archive/v_01.tar.gz',    # I explain this later on
    keywords = ['Hello', 'MEANINGFULL', 'KEYWORDS'],
    classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)

