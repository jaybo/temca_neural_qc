from setuptools import setup, find_packages

setup(name='temca-neural-qc',
      version='0.3',
      description='Scores the QC images from TEMCA',
      url='https://github.com/jaybo/temca_neural_qc',
      author='Jay Borseth',
      author_email='jayb@alleninstitute.org',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'temca_neural_qc = temca_neural_qc.temca_neural_qc:main',
          ],
      },
      install_requires=[
          'opencv-python',
          'tensorflow',
          'scikit-learn',
          'h5py',
          'numpy>1.16'
      ],)
