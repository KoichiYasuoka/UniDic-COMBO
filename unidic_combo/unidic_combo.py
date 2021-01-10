#! /usr/bin/python3 -i
# coding=utf-8

import os
PACKAGE_DIR=os.path.abspath(os.path.dirname(__file__))
combo_parser=None

class ComboAPI(object):
  def __init__(self,UniDic):
    self.lemmaform=False
    self.ipadic=(UniDic=="ipadic")
  def __call__(self,conllu):
    from unidic_combo.data import Token,Sentence,sentence2conllu
    u=[]
    e=[]
    for s in conllu.split("\n"):
      if s=="" or s.startswith("#"):
        if e!=[]:
          u.extend(combo_parser([Sentence(tokens=e)]))
          e=[]
      else:
        t=s.split("\t")
        xpos=t[4]
        if self.ipadic:
          if xpos.startswith("記号"):
            xpos="補助"+xpos
        if self.lemmaform:
          e.append(Token(id=int(t[0]),token=t[2],lemma=t[1],upostag=t[3],xpostag=xpos,misc=t[9]))
        else:
          e.append(Token(id=int(t[0]),token=t[1],lemma=t[2],upostag=t[3],xpostag=xpos,misc=t[9]))
    for s in u:
      for t in s.tokens:
        if self.lemmaform:
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
        if t.head==0 or t.head==t.id:
          t.head=0
          t.deprel="root"
    return "".join([sentence2conllu(s,False).serialize() for s in u])

class ComboRevAPI(ComboAPI):
  def __init__(self,UniDic):
    self.lemmaform=True
    self.ipadic=(UniDic=="ipadic")

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
    nlp.tokenizer.model.udpipe=ComboRevAPI(UniDic) if LemmaAsForm else ComboAPI(UniDic)
  nlp.tokenizer.model.model=m
  return nlp

