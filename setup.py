from setuptools import setup

setup(name='larnn',
      version='0.1',
      packages=['larnn'],
      zip_safe=False,
      author="Signe Moe, Filippo Remonato, Camilla Sterud",
      author_email="camilla.sterud@sintef.no",
      description="Implementation of the linear antisymmetric RNN cell as suggested in Moe et al. 2020.",
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: MIT License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.6',
      url='https://github.com/SINTEF/Linear-Antisymmetric-RNN',
      license='Apache License 2.0',
      install_requires=['tensorflow>=2.1.0', 'numpy>=1.19.0']
)
