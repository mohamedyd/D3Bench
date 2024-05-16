from setuptools import find_packages, setup
import os

# Define variables 
project_name = 'd3-benchmark'
author = 'Rieke Mueller'
email = 'rieke.mueller@softwareag.com'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open(os.path.join(os.path.dirname(__file__), 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line]

setup(
    name=project_name,
    packages=find_packages(),
    version='0.1.0',
    description='Source code of the thesis work carried out at Software AG',
    author=author,
    author_email=email,
    license='MIT',
    keywords='data quality, data distribution drift, ML pipelines',
    long_description=read('README.md'),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 1 - Planning"
    ]
)
