import setuptools,subprocess
subprocess.run(["./mkmodel.sh"])

setuptools.setup(
  name="unidic_combo",
  version="0.7.7",
  packages=setuptools.find_packages(),
  install_requires=[
    "unidic2ud>=2.6.2",
    "allennlp>=1.2.0",
    "torch>=1.6.0",
    "absl-py>=0.9.0",
    "conllu>=2.3.2",
    "dataclasses-json>=0.5.2",
    "requests>=2.23.0",
    "overrides>=3.1.0",
    "fugashi>=1.0.0",
    "ipadic>=1.0.0"
  ],
  python_requires=">=3.6",
  package_data={
    "unidic_combo":["./combo-*.tar.gz"]
  }
)
