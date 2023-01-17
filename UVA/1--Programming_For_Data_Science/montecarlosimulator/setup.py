from setuptools import setup

setup(
    name='montecarlosimulator',
    version='0.1.0',
    author='Tom Lever',
    author_email='tsl2b@virginia.edu',
    packages=['montecarlosimulator'],
    url='https://github.com/tslever/montecarlosimulator',
    license='MIT License',
    description='A Python package offering a Monte-Carlo Simulator',
    long_description=open('README.md').read(),
    install_requires=[],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10'
    ]
)
