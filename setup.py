import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="eqtm",
    version="1.5.0",
    author="Stefano Campanella",
    author_email="scampanella@inogs.it",
    description="Earthquake detection in continuous waveform data using template matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanocampanella/EQTM",
    packages=setuptools.find_packages(),
    scripts=['bin/eqtm-scan', 'bin/eqtm-avro2parquet', 'bin/eqtm-parquet2legacy', 'bin/eqtm-filter', 'bin/eqtm-select'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'obspy',
        'numpy',
        'scipy',
        'pandas',
        'bottleneck',
        'fastavro',
        'python-snappy',
        'fastparquet',
        'tqdm',
        'psutil'
    ],
    extras_require={'GPU acceleration': ['cupy']},
    include_package_data=True,
    python_requires='>=3.8',
)
