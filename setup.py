import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(name='IntelliMaint',
		version='0.0.1',
		author='IPTLP0032',
		author_email="author@example.com",
		description="A prognostics package by IPT",
		install_requires=['sklearn', 'GPy', 'minisom', 'scipy', 'matplotlib', 'numpy>=1.16.1'],
		packages=setuptools.find_packages(),
		package_data={
		'IntelliMaint.numpy_data': ['*']
		},
		python_requires='>=3.5',
	)
