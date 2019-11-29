from setuptools import find_packages, setup


# Read requirements file
REQUIREMENTS_FILE = 'requirements.txt'
with open(REQUIREMENTS_FILE) as f:
    REQUIREMENTS = f.read().splitlines()
print('Found the following requirements to be installed from {}:\n  {}'
      .format(REQUIREMENTS_FILE, '\n  '.join(REQUIREMENTS)))

# Collect packages
PACKAGES = find_packages(exclude=('tests', 'semi'))
print('Found the following packages to be created:\n  {}'
      .format('\n  '.join(PACKAGES)))

# Get long description from README
with open('README.md', 'r') as readme:
    LONG_DESCRIPTION = readme.read()

# Setup the package
setup(
    name='semi',
    version='0.0.1',
    packages=PACKAGES,
    python_requires='>=3.7.0',
    install_requires=REQUIREMENTS,
    url='https://github.com/JakobHavtorn/vae',
    author='Jakob Havtorn',
    description='Semi-supervised learning in PyTorch',
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
)