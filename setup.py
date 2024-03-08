from setuptools import setup

setup(
  name = 'chi-index',
  packages = ['chi_index','chi_index.utils'],
  version = '2.1',
  license='MIT',
  description = 'External Clustering Validation Chi index',
  long_description=open('README.md', 'r').read(),
  long_description_content_type='text/markdown',
  author = 'José María Luna-Romera',
  author_email = 'josemarialunaromera@gmail.com',
  url = 'https://github.com/josemarialuna/ChiIndex',
  download_url = 'https://github.com/josemarialuna/ChiIndex/tarball/0.1',
  keywords = ['chi index','cvi','clustering','machine learning'],
  install_requires=[
          'scikit-learn',
          'scipy',
          'numpy',
          'pandas',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)