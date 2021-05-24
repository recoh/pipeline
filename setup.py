from setuptools import setup

setup(
    name='pipeline',
    version='0.1',
    description='Image processing and quality control for abdominal MRI in the UK Biobank',
    license='MIT',
    author='Brandon Whitcher <b.whitcher@westminster.ac.uk>, Nicolas Basty <n.basty@westminster.ac.uk>',
    url='https://github.com/recoh',
    packages=['pipeline', 'cli', 'models'],
    scripts=['scripts/r-coh.py'],
    python_requires='>=3.3, <=3.7'
)
