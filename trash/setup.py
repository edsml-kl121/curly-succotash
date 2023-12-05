from setuptools import setup, find_packages

setup(
    name='tools',
    version='0.1.0',  # Alternatively, use `use_scm_version={"write_to": "tools/_version.py"}` if you have `setuptools_scm` configured
    author='Kandanai Leenutaphong',
    author_email='Kandanai.Leenutaphong@ibm.com',
    description="Environments for dhipaya",
    long_description="Environment for dhipaya",
    long_description_content_type='text/markdown',  # This is important if your long description is in Markdown
    # url='https://github.com/',  # Replace with your own repository URL
    packages=find_packages(),
    install_requires=[],
    python_requires='3.11',  # Adjust depending on your needs
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Change the license as needed
        'Operating System :: OS Independent',
    ],
    # include_package_data=True,  # Uncomment if you have package data to include
    # package_data={"package_name": ["data_files/*"]},  # Specify package data if needed
)