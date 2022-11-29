from setuptools import setup, find_packages

setup(name='Metric Learning Layers',
      version='0.1.0',
      description='A simple PyTorch package that includes the most common metric learning layers',
      author='Robert Müller',
      author_email='robert.mueller1990@googlemail.com',
      url='https://github.com/romue404/metric_learning_layers',
      license='MIT',
      packages=find_packages(exclude=['examples']),
      keywords=[
            'metric learning',
            'artificial intelligence',
            'pytorch',
            'separability',
            'large margin'
      ],
      classifiers=[
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3.6',
      ],
      install_requires=['torch>=1.6', 'typing']
)