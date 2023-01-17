from setuptools import setup

setup(
    name='booklover',
    version='0.1.0',
    author='Tom Lever',
    author_email='tsl2b@virginia.edu',
    packages=['booklover'],
    url='https://github.com/tslever/booklover',
    license='MIT License',
    description='A Python package offering class BookLover',
    long_description=open('README.md').read(),
    install_requires=['pandas >= 1.5.1'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10'
    ]
)
