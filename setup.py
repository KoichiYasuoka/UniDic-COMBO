import os,setuptools,subprocess
if os.path.isfile("dummy.tar.gz"):
  package_data={"unidic_combo":["./mkmodel.sh"]}
else:
  subprocess.run(["unidic_combo/mkmodel.sh"])
  package_data={"unidic_combo":["./mkmodel.sh","./combo-*.tar.gz"]}
with open("README.md","r",encoding="utf-8") as r:
  long_description=r.read()
URL="https://github.com/KoichiYasuoka/UniDic-COMBO"

setuptools.setup(
  name="unidic_combo",
  version="0.8.4",
  description="UniDic2UD + COMBO-pytorch wrapper for spaCy",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url=URL,
  author="Koichi Yasuoka",
  author_email="yasuoka@kanji.zinbun.kyoto-u.ac.jp",
  license="GPL",
  keywords="NLP Japanese spaCy",
  packages=setuptools.find_packages(),
  install_requires=[
    "unidic2ud>=2.6.6",
    "spacy>=2.2.2",
    "allennlp>=1.2.0",
    "torch>=1.6.0",
    "absl-py>=0.9.0",
    "conllu>=2.3.2",
    "dataclasses-json>=0.5.2",
    "requests>=2.23.0",
    "overrides>=3.1.0",
    "fugashi>=1.0.1",
    "ipadic>=1.0.0"
  ],
  python_requires=">=3.6",
  package_data=package_data,
  classifiers=[
    "License :: OSI Approved :: GNU General Public License (GPL)",
    "Programming Language :: Python :: 3",
    "Operating System :: OS Independent",
    "Topic :: Text Processing :: Linguistic",
    "Natural Language :: Japanese"
  ],
  project_urls={
    "COMBO-pytorch":"https://github.com/ipipan/combo",
    "Source":URL,
    "Tracker":URL+"/issues",
  }
)
