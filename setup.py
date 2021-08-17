import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="correlation-detector",
    version="1.1.0",
    author="Stefano Campanella",
    author_email="scampanella@inogs.it",
    description="Earthquake detection in continuous waveform data using template matching",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stefanocampanella/correlation-detector",
    packages=setuptools.find_packages(),
    scripts=['bin/correlation-detector', 'bin/detections-stats', 'bin/cat-detections'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'bottleneck',
        'tqdm',
        'psutil',
        'fastavro',
        'python-snappy',
        'obspy'
    ],
    extras_require={'GPU acceleration': ['cupy']},
    include_package_data=True,
    python_requires='>=3.8',
)
