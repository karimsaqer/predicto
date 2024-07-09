from setuptools import setup, find_packages

setup(
    name='vpredicto',
    version='0.1.13',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',
        'torchvision>=0.8.1',
        'pytorch-lightning>=1.0.0',
        'scikit-image>=0.17.2',
        'numpy>=1.19.2',
    ],
    author='Team 18',
    author_email='Karim.Saqer01@eng-st.cu.edu.eg',
    description='A library for video frame prediction using PredRNN++, MIM, and Causal LSTM.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/karimsaqer/predicto',
    python_requires='>=3.6',
)
