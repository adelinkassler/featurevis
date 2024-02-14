from setuptools import find_packages, setup


# Function to read the contents of the requirements.txt file
def read_requirements():
    with open('requirements.txt', 'r') as req:
        return req.read().splitlines()

setup(
    name='vision',
    version='0.1.0',
    author='Timaeus',
    author_email='contact@timaeus.co',
    description='A research project on Computer Vision (CV) models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/timaeus-research/vision',
    packages=find_packages(),
    install_requires=read_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
