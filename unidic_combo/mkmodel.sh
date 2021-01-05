#! /bin/sh
for M in japanese japanese-rev japanese-small japanese-small-rev
do if [ -s unidic_combo/combo-$M.tar.gz ]
   then continue
   elif [ -s model/combo-$M.tar.gz.1 ]
   then cat model/combo-$M.tar.gz.[1-9] > unidic_combo/combo-$M.tar.gz
   elif [ -s UniDic-COMBO/model/combo-$M.tar.gz.1 ]
   then cat UniDic-COMBO/model/combo-$M.tar.gz.[1-9] > unidic_combo/combo-$M.tar.gz
   elif [ -s ja_gsd_modern.conllu ]
   then case "$M" in
        *-rev) sed 's/^\([1-9][0-9]	\)\([^	]*	\)\([^	]*	\)/\1\3\2/' ja_gsd_modern.conllu > ja_rev.conllu
               C=ja_rev.conllu ;;
        *) C=ja_gsd_modern.conllu ;;
        esac
        case "$M" in
        *-small*) B='' ;;
        *) B='--pretrained_transformer_name cl-tohoku/bert-base-japanese-whole-word-masking' ;;
        esac
        python3 -m unidic_combo.main --mode train --cuda_device 0 --num_epochs 100 $B --training_data_path $C --targets deprel,head,upostag --features token,char,xpostag,lemma
        cp `ls -1t /tmp/allennlp*/model.tar.gz | head -1` unidic_combo/combo-$M.tar.gz
        mkdir -p model
        split -a 1 -b 83886080 --numeric-suffixes=1 unidic_combo/combo-$M.tar.gz model/combo-$M.tar.gz.
   else git clone --depth=1 https://github.com/KoichiYasuoka/UniDic-COMBO
        cat UniDic-COMBO/model/combo-$M.tar.gz.[1-9] > unidic_combo/combo-$M.tar.gz
   fi
done
exit 0
