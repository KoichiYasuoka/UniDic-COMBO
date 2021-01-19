#! /bin/sh
for M in japanese japanese-rev japanese-small japanese-small-rev
do if [ -s combo-$M.tar.gz ]
   then continue
   elif [ -s combo-$M.tar.gz.1 ]
   then cat combo-$M.tar.gz.[1-9] > combo-$M.tar.gz
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
        cp `ls -1t /tmp/allennlp*/model.tar.gz | head -1` combo-$M.tar.gz
        split -a 1 -b 83886080 --numeric-suffixes=1 combo-$M.tar.gz combo-$M.tar.gz.
   fi
done
exit 0
