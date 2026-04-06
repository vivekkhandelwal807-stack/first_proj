from setuptools import find_packages,setup
from typing import List

def get_requirement(file_path:str)-> List[str]:

    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()

    return requirements












setup(

    name='mlproject',
    version='0.0.1',
    author='vivek',
    packages=find_packages(),
    install_requires=get_requirement('requirements'),


)