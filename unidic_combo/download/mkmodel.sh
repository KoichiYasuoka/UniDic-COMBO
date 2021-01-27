#! /bin/sh
if [ $# -eq 0 ]
then set combo-japanese combo-japanese-rev combo-japanese-small combo-japanese-small-rev
fi
for M
do if [ -s $M.tar.gz ]
   then continue
   elif [ -s $M.tar.gz.1 ]
   then cat $M.tar.gz.[1-9] > $M.tar.gz
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
        cp `ls -1t /tmp/allennlp*/model.tar.gz | head -1` $M.tar.gz
        split -a 1 -b 83886080 --numeric-suffixes=1 $M.tar.gz $M.tar.gz.
   fi
done
ls -ltr *.tar.gz | awk '{printf("%s %d\n",$NF,$5)}' > filesize.txt
exit 0
