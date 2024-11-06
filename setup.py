from setuptools import find_packages, setup
from typing import List

hypendote='-e .'
def get_requirements(file_path:str)-> List[str]:
    # This function will return the list of requiremnts 
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n"," ") for req in requirements]
        if hypendote in requirements:
            requirements.remove(hypendote)
    return requirements

setup(
name='mlprojects',
version='0.0.1',
author=' ',
packages=find_packages(),
install_requires=get_requirements('Requirements.txt')





)