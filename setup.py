from setuptools import setup, find_packages

setup(
    name='dynamic-signs-recognition-libras-alphabet-cnn',
    version='0.1',
    packages=find_packages(where='src'), 
    package_dir={'': 'src'},
    install_requires=[
        # Manipulação de arquivos e dados
        'numpy',
        'pandas',
        'opencv-python',
        'matplotlib',
        'scikit-learn',
        
        # Processamento de vídeos
        'ffmpeg',
        'imageio[ffmpeg]',
        
        # Deep Learning
        'tensorflow',
        'keras',
        'torch',
        'torchvision',
        
        # Notebook e visualização
        'jupyter',
        'notebook',

        #Gráficos
        'seaborn'
    ],
    python_requires='>=3.6',
)