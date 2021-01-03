#! /bin/sh
rm -fr build dist *.egg-info
python3 setup.py sdist
git status
twine upload --repository pypi dist/*
exit 0
