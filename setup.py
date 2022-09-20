from setuptools import setup
from codecs import open
from os import path

current = path.abspath(path.dirname(__file__))
INSTALL_REQUIRES = [
    'matplotlib>=3.5.1',
]
EXTRAS_REQUIRE = {}

with open(path.join(current, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='myforestplot', 
    packages=['myforestplot'], 
    license='MIT', 

    author='toshiakiasakura', 
    author_email='wordpress.akitoshi@gmail.com', 
    url='https://toshiakiasakura.github.io/myforestplot', 

    description='A flexibly customizable Python tool to create a forest plot.', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    keywords='forestplot odds-ratio relative-risk meta-analysis', 

    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,

    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Framework :: Matplotlib',
    ], 
)

