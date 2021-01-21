#! /usr/bin/python3 -i
# coding=utf-8

import os
PACKAGE_DIR=os.path.abspath(os.path.dirname(__file__))
DOWNLOAD_DIR=os.path.join(PACKAGE_DIR,"download")
MODEL_URL="https://raw.githubusercontent.com/KoichiYasuoka/UniDic-COMBO/master/unidic_combo/download/"
filesize={}
with open(os.path.join(DOWNLOAD_DIR,"filesize.txt"),"r") as f:
  r=f.read()
for t in r.split("\n"):
  s=t.split()
  if len(s)==2:
    filesize[s[0]]=int(s[1])

import time
tm=time.time()
fs=ft=0
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
          u.append(Sentence(tokens=e))
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
    u=combo_parser(u)
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
    f=os.path.join(DOWNLOAD_DIR,m)
    try:
      s=os.path.getsize(f)
      if filesize[m]!=s:
        s=-1
    except:
      s=-1
    if s<0:
      download(MODEL_URL,m,DOWNLOAD_DIR)
    combo_parser=unidic_combo.predict.COMBO.from_pretrained(f)
    nlp.tokenizer.model.udpipe=ComboRevAPI(UniDic) if LemmaAsForm else ComboAPI(UniDic)
  nlp.tokenizer.model.model=m
  return nlp

def progress(block_count,block_size,total_size):
  t=time.time()
  p=100.0*(block_count*block_size+fs)/ft
  if p<1:
    t=-1
  elif p>=100:
    p=100
    t-=tm
  else:
    t=(t-tm)*(100-p)/p
  b=int(p/2)
  if b==50:
    s="="*50
  else:
    s=("="*b)+">"+(" "*(49-b))
  if t<0:
    u="   "
  elif t<3600:
    u=time.strftime("%M:%S   ",time.gmtime(t))
  elif t<86400:
    u=time.strftime("%H:%M:%S   ",time.gmtime(t))
  else:
    u=time.strftime("%d+%H:%M:%S   ",time.gmtime(t))
  print("\r ["+s+"] "+str(int(p))+"% "+u,end="")

def download(url,file,dir="."):
  import urllib.request
  global fs,ft,tm
  t=os.path.join(dir,"filesize.txt")
  urllib.request.urlretrieve(url+"filesize.txt",t)
  with open(t,"r") as f:
    r=f.read()
  ft=0
  for t in r.split("\n"):
    s=t.split()
    if len(s)==2:
      if s[0]==file:
        ft=int(s[1])
  tm=time.time()
  with open(os.path.join(dir,file),"wb") as f:
    fs=0
    i=1
    while fs<ft:
      g,h=urllib.request.urlretrieve(url+file+"."+str(i),reporthook=progress)
      with open(g,"rb") as r:
        q=r.read()
      f.write(q)
      fs+=len(q)
      i+=1
  print("")

