from setuptools import setup

setup(name='temca_neural_qc',
      version='0.1',
      description='Scores the QC images from TEMCA',
      url='https://github.com/jaybo/temca_neural_qc',
      author='Jay Borseth',
      author_email='jayb@alleninstitute.org',
      license='MIT',
      packages=['temca_neural_qc'],
      zip_safe=False,
      install_requires=[
          'tensorflow-gpu',
      ],)