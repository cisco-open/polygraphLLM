from setuptools import find_packages, setup


def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


requirements = parse_requirements('requirements.txt')


setup(
    name='halludetector',
    version='0.0.8',
    author='Mihai Onofrei',
    author_email='monofrei@cisco.com',
    url='https://github.com/Mihai-Onofrei/Hallucination-detector',  # Replace with your project's URL
    license='MIT',  # Choose an appropriate license
    description="Hallucination detection package",
    long_description=open('README.rst').read(),
    include_package_data=True,
    packages=find_packages(),
    package_data={
        '': ['*.json'],  # Include all JSON files in the package
        'logos': ['*.png'],
        'js': ['*.js'],
        'txt': ['*.txt']
    },
    install_requires=requirements
)
