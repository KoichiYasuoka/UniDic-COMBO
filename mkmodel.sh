#! /bin/sh
if [ -s unidic_combo/combo-japanese.tar.gz ]
then :
elif [ -s model/combo-japanese.tar.gz.1 ]
then cat model/combo-japanese.tar.gz.[1-9] > unidic_combo/combo-japanese.tar.gz
elif [ -s UniDic-COMBO/model/combo-japanese.tar.gz.1 ]
then cat UniDic-COMBO/model/combo-japanese.tar.gz.[1-9] > unidic_combo/combo-japanese.tar.gz
elif [ -s ja_gsd_modern.conllu ]
then python3 -m unidic_combo.main --mode train --cuda_device 0 --num_epochs 100 --pretrained_transformer_name cl-tohoku/bert-base-japanese-whole-word-masking --training_data_path ja_gsd_modern.conllu --targets deprel,head,upostag --features token,char,xpostag,lemma
     cp -i `ls -1t /tmp/allennlp*/model.tar.gz | head -1` unidic_combo/combo-japanese.tar.gz
     mkdir -p model
     split -a 1 -b 83886080 --numeric-suffixes=1 unidic_combo/combo-japanese.tar.gz model/combo-japanese.tar.gz.
else git clone --depth=1 https://github.com/KoichiYasuoka/UniDic-COMBO
     cat UniDic-COMBO/model/combo-japanese.tar.gz.[1-9] > unidic_combo/combo-japanese.tar.gz
fi

if [ -s unidic_combo/combo-japanese-rev.tar.gz ]
then :
elif [ -s model/combo-japanese-rev.tar.gz.1 ]
then cat model/combo-japanese-rev.tar.gz.[1-9] > unidic_combo/combo-japanese-rev.tar.gz
elif [ -s UniDic-COMBO/model/combo-japanese-rev.tar.gz.1 ]
then cat UniDic-COMBO/model/combo-japanese-rev.tar.gz.[1-9] > unidic_combo/combo-japanese-rev.tar.gz
elif [ -s ja_gsd_modern.conllu ]
then sed 's/^\([1-9][0-9]	\)\([^	]*	\)\([^	]*	\)/\1\3\2/' ja_gsd_modern.conllu > ja_rev.conllu
     python3 -m unidic_combo.main --mode train --cuda_device 0 --num_epochs 100 --pretrained_transformer_name cl-tohoku/bert-base-japanese-whole-word-masking --training_data_path ja_rev.conllu --targets deprel,head,upostag --features token,char,xpostag,lemma
     cp -i `ls -1t /tmp/allennlp*/model.tar.gz | head -1` unidic_combo/combo-japanese-rev.tar.gz
     mkdir -p model
     split -a 1 -b 83886080 --numeric-suffixes=1 unidic_combo/combo-japanese-rev.tar.gz model/combo-japanese-rev.tar.gz.
else git clone --depth=1 https://github.com/KoichiYasuoka/UniDic-COMBO
     cat UniDic-COMBO/model/combo-japanese-rev.tar.gz.[1-9] > unidic_combo/combo-japanese-rev.tar.gz
fi
exit 0
