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
  u=combo_parser(d)
  for s in u:
    for t in s.tokens:
      d=t.deprel
      if d=="root":
        if t.head!=0:
          t.deprel="advcl" if t.head>t.id else "parataxis"
      elif d=="advmod":
        t.upostag="ADV"
      elif d=="amod":
        t.upostag="ADJ"
      elif d=="aux" or d=="cop":
        t.upostag="AUX"
      elif d=="det":
        t.upostag="DET"
      elif d=="nummod":
        t.upostag="NUM"
  return "".join([sentence2conllu(s,False).serialize() for s in u])

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
      d=t.deprel
      if d=="root":
        if t.head!=0:
          t.deprel="advcl" if t.head>t.id else "parataxis"
      elif d=="advmod":
        t.upostag="ADV"
      elif d=="amod":
        t.upostag="ADJ"
      elif d=="aux" or d=="cop":
        t.upostag="AUX"
      elif d=="det":
        t.upostag="DET"
      elif d=="nummod":
        t.upostag="NUM"
  return "".join([sentence2conllu(s,False).serialize() for s in u])

def load(UniDic=None,BERT=True,LemmaAsForm=None):
  global combo_parser
  import unidic2ud.spacy
  if UniDic==None:
    UniDic="ipadic"
  nlp=unidic2ud.spacy.load(UniDic,None)
  m="combo-japanese.tar.gz" if BERT else "combo-japanese-small.tar.gz"
  if LemmaAsForm==None:
    LemmaAsForm=UniDic not in ["gendai","spoken","ipadic"]
  if LemmaAsForm:
    m=m.replace(".tar.gz","-rev.tar.gz")
  if nlp.tokenizer.model.model!=m:
    combo_parser=None
  if combo_parser==None:
    import unidic_combo.predict
    combo_parser=unidic_combo.predict.SemanticMultitaskPredictor.from_pretrained(os.path.join(PACKAGE_DIR,m))
    nlp.tokenizer.model.udpipe=ComboRevAPI if LemmaAsForm else ComboAPI
  nlp.tokenizer.model.model=m
  return nlp

