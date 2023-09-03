from setuptools import setup, find_packages


setup(name='SPAL',
      version='1.0.0',
      description='Support Pair Active Learning for Person Re-identification',
      author='Dapeng Jin',
      author_email='dpjin@outlook.com',
      url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Active Learning',
          'Person Re-identification'
      ])
