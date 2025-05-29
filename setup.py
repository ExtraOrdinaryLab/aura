from setuptools import find_packages, setup


setup(
    name='aura', 
    version='0.0.1', 
    description='Open-source library of Atypical Speech Recognition', 
    author='Yang Wang', 
    author_email='yangwang4work@gmail.com', 
    package_dir={'': 'src'}, 
    packages=find_packages('src'), 
    python_requires='>=3.10.0', 
    keywords='atypical'
)