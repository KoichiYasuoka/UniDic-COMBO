[![Current PyPI packages](https://badge.fury.io/py/unidic-combo.svg)](https://pypi.org/project/unidic-combo/)

# UniDic-COMBO

[UniDic2UD](https://github.com/KoichiYasuoka/UniDic2UD) + [COMBO-pytorch](https://github.com/ipipan/combo) wrapper for [spaCy](https://spacy.io)

## Basic Usage

```py
>>> import unidic_combo
>>> nlp=unidic_combo.load("kindai")
>>> doc=nlp("澤山居つた兄弟が一疋も見えぬ")
>>> print(unidic_combo.to_conllu(doc))
# text = 澤山居つた兄弟が一疋も見えぬ
1	澤山	沢山	ADV	副詞	_	2	advmod	_	SpaceAfter=No|Translit=タクサン
2	居つ	居る	VERB	動詞-非自立可能	_	4	acl	_	SpaceAfter=No|Translit=オッ
3	た	た	AUX	助動詞	_	2	aux	_	SpaceAfter=No|Translit=タ
4	兄弟	兄弟	NOUN	名詞-普通名詞-一般	_	9	nsubj	_	SpaceAfter=No|Translit=キョウダイ
5	が	が	ADP	助詞-格助詞	_	4	case	_	SpaceAfter=No|Translit=ガ
6	一	一	NUM	名詞-数詞	_	7	nummod	_	SpaceAfter=No|Translit=イチ
7	疋	匹	NOUN	接尾辞-名詞的-助数詞	_	9	obl	_	SpaceAfter=No|Translit=ピキ
8	も	も	ADP	助詞-係助詞	_	7	case	_	SpaceAfter=No|Translit=モ
9	見え	見える	VERB	動詞-一般	_	0	root	_	SpaceAfter=No|Translit=ミエ
10	ぬ	ず	AUX	助動詞	_	9	aux	_	SpaceAfter=No|Translit=ヌ

>>> import deplacy
>>> deplacy.render(doc,Japanese=True)
澤山 ADV  <══╗     advmod(連用修飾語)
居つ VERB ═╗═╝<╗   acl(連体修飾節)
た   AUX  <╝   ║   aux(動詞補助成分)
兄弟 NOUN ═╗═══╝<╗ nsubj(主語)
が   ADP  <╝     ║ case(格表示)
一   NUM  <╗     ║ nummod(数量による修飾語)
疋   NOUN ═╝═╗<╗ ║ obl(斜格補語)
も   ADP  <══╝ ║ ║ case(格表示)
見え VERB ═╗═══╝═╝ ROOT(親)
ぬ   AUX  <╝       aux(動詞補助成分)

>>> from deplacy.deprelja import deprelja
>>> for b in unidic_combo.bunsetu_spans(doc):
...   for t in b.lefts:
...     print(unidic_combo.bunsetu_span(t),"->",b,"("+deprelja[t.dep_]+")")
...
澤山 -> 居つた (連用修飾語)
居つた -> 兄弟が (連体修飾節)
兄弟が -> 見えぬ (主語)
一疋も -> 見えぬ (斜格補語)
```

`unidic_combo.load(UniDic)` loads spaCy Language pipeline for UniDic2UD + COMBO-pytorch. Available `UniDic` options are:

* `UniDic="gendai"`: Use [現代書き言葉UniDic](https://unidic.ninjal.ac.jp/download#unidic_bccwj).
* `UniDic="spoken"`: Use [現代話し言葉UniDic](https://unidic.ninjal.ac.jp/download#unidic_csj).
* `UniDic="qkana"`: Use [旧仮名口語UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_qkana).
* `UniDic="kindai"`: Use [近代文語UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_kindai).
* `UniDic="kinsei"`: Use [近世口語（洒落本）UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_kinsei).
* `UniDic="kyogen"`: Use [中世口語（狂言）UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_kyogen).
* `UniDic="wakan"`: Use [中世文語（説話・随筆）UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_wakan).
* `UniDic="wabun"`: Use [中古和文UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_wabun).
* `UniDic="manyo"`: Use [上代（万葉集）UniDic](https://unidic.ninjal.ac.jp/download_all#unidic_manyo).
* `UniDic=None`: Use IPADic (default).

## Installation for Linux

```sh
pip3 install unidic_combo
```

## Installation for Cygwin64

Make sure to get `python37-devel` `python37-pip` `python37-cython` `python37-numpy` `python37-cffi` `gcc-g++` `mingw64-x86_64-gcc-g++` `gcc-fortran` `git` `curl` `make` `cmake` `libopenblas` `liblapack-devel` `libhdf5-devel` `libfreetype-devel` `libuv-devel` packages, and then:
```sh
curl -L https://raw.githubusercontent.com/KoichiYasuoka/UniDic-COMBO/master/cygwin64.sh | sh
```

## Installation for macOS

```sh
g++ --version
pip3 install unidic_combo --user
python3 -m spacy download en_core_web_sm --user
```

If you fail to install [Jsonnet](https://github.com/google/jsonnet), try below before installing UniDic-COMBO:

```sh
( echo '#! /bin/sh' ; echo 'exec gcc `echo $* | sed "s/-arch [^ ]*//g"`' ) > /tmp/clang
chmod 755 /tmp/clang
env PATH="/tmp:$PATH" pip3 install jsonnet --user
```

## Benchmarks

Results of [舞姬/雪國/荒野より-Benchmarks](https://colab.research.google.com/github/KoichiYasuoka/UniDic-COMBO/blob/master/benchmark.ipynb)

|[舞姬](https://github.com/KoichiYasuoka/UniDic2UD/blob/master/benchmark/maihime-benchmark.tar.gz)|LAS|MLAS|BLEX|
|---------------|-----|-----|-----|
|UniDic="kindai"|83.02|74.07|81.48|
|UniDic="qkana" |81.13|74.07|81.48|
|UniDic="kinsei"|75.93|69.09|72.73|

|[雪國](https://github.com/KoichiYasuoka/UniDic2UD/blob/master/benchmark/yukiguni-benchmark.tar.gz)|LAS|MLAS|BLEX|
|---------------|-----|-----|-----|
|UniDic="qkana" |87.50|82.35|78.43|
|UniDic="kinsei"|85.71|78.43|74.51|
|UniDic="kindai"|83.19|78.43|74.51|

|[荒野より](https://github.com/KoichiYasuoka/UniDic2UD/blob/master/benchmark/koyayori-benchmark.tar.gz)|LAS|MLAS|BLEX|
|---------------|-----|-----|-----|
|UniDic="qkana" |77.49|64.00|64.00|
|UniDic="kindai"|78.53|61.33|61.33|
|UniDic="kinsei"|69.79|60.53|57.89|

