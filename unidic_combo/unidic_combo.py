#! /usr/bin/python3 -i
# coding=utf-8

import os
PACKAGE_DIR=os.path.abspath(os.path.dirname(__file__))
combo_parser=None

def ComboAPI(conllu):
  from unidic_combo.data import Token,Sentence,sentence2conllu
  d=[]
  e=[]
  for s in conllu.split("\n"):
    if s=="" or s.startswith("#"):
      if e!=[]:
        d.append(Sentence(tokens=e))
        e=[]
    else:
      t=s.split("\t")
      e.append(Token(id=int(t[0]),token=t[1],lemma=t[2],upostag=t[3],xpostag=t[4],misc=t[9]))
  return "".join([sentence2conllu(s,False).serialize() for s in combo_parser(d)])

def ComboRevAPI(conllu):
  from unidic_combo.data import Token,Sentence,sentence2conllu
  d=[]
  e=[]
  for s in conllu.split("\n"):
    if s=="" or s.startswith("#"):
      if e!=[]:
        d.append(Sentence(tokens=e))
        e=[]
    else:
      t=s.split("\t")
      e.append(Token(id=int(t[0]),token=t[2],lemma=t[1],upostag=t[3],xpostag=t[4],misc=t[9]))
  u=combo_parser(d)
  for s in u:
    for t in s.tokens:
      t.token,t.lemma=t.lemma,t.token
  return "".join([sentence2conllu(s,False).serialize() for s in u])

def load(UniDic=None,LemmaAsForm=None):
  global combo_parser
  import unidic2ud.spacy
  if UniDic==None:
    UniDic="ipadic"
  nlp=unidic2ud.spacy.load(UniDic,None)
  if LemmaAsForm==None:
    LemmaAsForm=UniDic not in ["gendai","spoken","ipadic"]
  m="combo-japanese-rev.tar.gz" if LemmaAsForm else "combo-japanese.tar.gz"
  if nlp.tokenizer.model.model!=m:
    combo_parser=None
  if combo_parser==None:
    import unidic_combo.predict
    combo_parser=unidic_combo.predict.SemanticMultitaskPredictor.from_pretrained(os.path.join(PACKAGE_DIR,m))
    nlp.tokenizer.model.udpipe=ComboRevAPI if LemmaAsForm else ComboAPI
  nlp.tokenizer.model.model=m
  return nlp

