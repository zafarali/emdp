from setuptools import setup, find_packages
from emdp import __version__

base_requirements = ['numpy>=1.9.1']
extras = {
    'tests': ['gym', 'matplotlib'],
    'gym': ['gym']
}

setup(
    name='emdp',
    version=__version__,
    description='Easy MDPs',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/zafarali/emdp',
    author='Zafarali Ahmed',
    author_email='zafarali.ahmed@gmail.com',
    license='MIT',
    packages=find_packages(), 
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    python_requires='>=3.5',
    extras_require=extras,
    install_requires=base_requirements,
)

