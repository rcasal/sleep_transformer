from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='sleep_transformer',
      version='0.1',
      description='sleep transformer for classifying sleep stages in awake or asleep using heart rate',
      url='https://github.com/rcasal/sleep_transformer.git',
      author='Ramiro Casal',
      author_email='rcasal@conicet.gov.ar',
      license='MIT',
      packages=['sleep_transformer'],
      entry_points={},
      install_requires=[
          'numpy',
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3'],
      zip_safe=False)
