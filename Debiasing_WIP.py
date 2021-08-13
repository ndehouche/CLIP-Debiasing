import torch
from PIL import Image
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
import glob
import csv

images=glob.glob("Images/*.jpg")

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32') #Tokenizer for finetuning data



model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

data = []
ie = [] #Image embeddings, equal for all labels
we = [] #Word embeddings, for neutral labels
for image in images:
 image=Image.open(image).resize((200, 200), Image.NEAREST)
 inputs0 = processor(text=["male", "female"], images=image, return_tensors="pt", padding=True)
 inputs1 = processor(text=["he", "she"], images=image, return_tensors="pt", padding=True)
 inputs2 = processor(text=["father", "mother"], images=image, return_tensors="pt", padding=True)
 inputs3 = processor(text=["son", "daughter"], images=image, return_tensors="pt", padding=True)
 inputs4 = processor(text=["John", "Mary"], images=image, return_tensors="pt", padding=True)
 inputs5 = processor(text=["man", "woman"], images=image, return_tensors="pt", padding=True)
 inputs6 = processor(text=["boy", "girl"], images=image, return_tensors="pt", padding=True)
 inputs7 = processor(text=["himself", "herself"], images=image, return_tensors="pt", padding=True)
 inputs8 = processor(text=["guy", "gal"], images=image, return_tensors="pt", padding=True)
 inputs9 = processor(text=["his", "her"], images=image, return_tensors="pt", padding=True)
 for i in range(10):
  globals()['outputs%s' % i] = model(**globals()['inputs%s' % i])
  ie.append(outputs0.image_embeds.detach().numpy()) 
  #Text embeddings 
  globals()['te%s' % i]=globals()['outputs%s' % i].text_embeds 
  #Differences 
  globals()['diff%s' % i]=globals()['te%s' % i][0]-globals()['te%s' % i][1]
  data.append(globals()['diff%s' % i].detach().numpy())
data = np.array(data)
ie = np.array(ie)
data=data / np.sqrt(np.sum(data**2))
df = pd.DataFrame(data=data)
df.to_csv('out.csv',index=False)
df = pd.DataFrame(data=data)
from sklearn.decomposition import PCA
pca=PCA(n_components=10)
pca.fit(df)
print(pca.explained_variance_ratio_)
g=pca.components_[0]
g=g / np.linalg.norm(g)
direct_bias=0

#Pairs of labels that are supposed to be gender-neutral for this application
pair0 = processor(text=["rich", "poor"], images=image, return_tensors="pt", padding=True)
pair1 = processor(text=["attractive", "unattractive"], images=image, return_tensors="pt", padding=True)


#Direct bias in word embeddings
direct_bias=0
for i in range(2):
 for j in range(2):
  globals()['out%s' % i] = model(**globals()['pair%s' % i])
  we=globals()['out%s' % i].text_embeds.detach().numpy()[j]
  unit_vector = we / np.linalg.norm(we)
  direct_bias=direct_bias+abs(np.dot(g, unit_vector))
direct_bias=direct_bias/4
print(direct_bias)


beta = []
#Indirect bias in image-text similarity 
for i in range(10000):
 for j in range(2):
  for k in range(2):
   globals()['out%s' % j] = model(**globals()['pair%s' % j])
   te=globals()['out%s' % j].text_embeds.detach().numpy()[k]
   te = te / np.linalg.norm(te)
   tg=np.dot(g, te)*g
   ie[i]=ie[i]/ np.linalg.norm(ie[i])
   ig=np.dot(g, ie[i][0])*g
   t_orthog=te-tg
   i_orthog=ie[i][0]-ig
   ratio=(np.dot(te,ie[i][0])-(np.dot(t_orthog,i_orthog)/(np.linalg.norm(t_orthog)*np.linalg.norm(i_orthog))))/np.dot(te,ie[i][0])
   beta.append(ratio)
beta = np.array(beta)   
print(beta)   






