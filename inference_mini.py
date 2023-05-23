from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image
import torch

import glob

import nltk
#nltk.download('punkt')

from nltk import word_tokenize

from transformers import GPT2TokenizerFast, AutoTokenizer

#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
#tokenizer = AutoTokenizer.from_pretrained("gpt-medium", use_fast=False)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/meta_ori2/global_step29100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
model.eval()
#"../../check/global_step19945/mp_rank_00_model_states.pt"

#val_df=pd.read_csv('val.csv')
#train_df=pd.read_csv('train.csv')
train_df=pd.read_csv('mini/mini.csv')
val_df=pd.read_csv('mini/mini.csv')

'''

path_lst=glob.glob("/local1/monajati/kaggle/*") 
print(path_lst[0])
print(path_lst[1])
'''

#'n01532829 n01558993'
#'n01843383 n01855672'
"n01532829_10006.JPEG"
"n01532829_1097.JPEG"
"n01558993_10268.JPEG"
"n01558993_1126.JPEG"

'n02099601_100.JPEG'
'n02099601_10165.JPEG'

c=0
for i in range(100):
    
    print("i",i)

    paths1=glob.glob("/local1/monajati/kaggle/n01843383/*")
    
    ind = random.randint(0,len(val_df))
    im_id=val_df['id'][ind]
    #ImageNet-Mini/M_images/
    paths1=glob.glob("/home/monajati/main/metaVL/magma/mini/ImageNet-Mini/images/"+str(im_id)+"/*")
    #paths1=glob.glob("/home/monajati/main/metaVL/magma/mini/")
    
    print(paths1)
    
    if len(paths1)<2:
        continue

    path1_1=paths1[0]
    #path1_1='/local1/monajati/kaggle/n01532829/n01532829_10006.JPEG'
    
    answer1=val_df['label'][ind].replace("_", " " )
    words = word_tokenize(answer1)
    if len(words)>1:
        answer1 = words[-1]
    #answer1='bird'
    
    #path1_2='/local1/monajati/kaggle/n01532829/n01532829_1097.JPEG'
    path1_2=paths1[1]
    
    #im_id2=val_df['im_id'][ind2]
    
    paths2=glob.glob("/local1/monajati/kaggle/n01855672/*")
    
    ind = random.randint(0,len(val_df))
    im_id=val_df['id'][ind]
    
    paths2=glob.glob("/home/monajati/main/metaVL/magma/mini/ImageNet-Mini/images/"+str(im_id)+"/*")
    
    print(paths2)
    
    if len(paths2)<2:
        continue
    
    path2_1=paths2[0]
    #path2_1='/local1/monajati/kaggle/n02099601/n02099601_100.JPEG'
    answer2=val_df['label'][ind].replace("_", " ")
    
    words = word_tokenize(answer2)
    if len(words)>1:
        answer2 = words[-1]
    #answer2='dog'
    
    path2_2=paths2[1]
    #path2_2='/local1/monajati/kaggle/n02099601/n02099601_10165.JPEG'
    
    prompt= 'Answer with ' + answer1 + ' or ' + answer2 + '.'
    
    inputs =[prompt ,
             ImageInput(path1_1),
    'Question: what is this? Answer: ' + answer1 ,
             prompt ,
             ImageInput(path2_1),
    'Question: what is this? Answer: '+ answer2 ,
             prompt ,
             ImageInput(path1_2),
    'Question: what is this? Answer: ' + answer1 ,
             prompt ,
             ImageInput(path2_2),
    'Question: what is this? Answer:' 
    ]
    
    inputs =[prompt, ImageInput(path1_1),
    'This is a ' + answer1 + '.' ,
             ImageInput(path2_1),
    'This is a '+ answer2  + '.',
             ImageInput(path1_2),
    'This is a '+ answer1  + '.',
             ImageInput(path2_2),
    'This is a' 
    ]

    '''
    ImageInput(path1_2),
    'Q: This is a ? ' + answer1 + '.',
             ImageInput(path2_2),
    'Q: This is a ? ' + answer2 + '.',
    '''
    
    print(inputs)

   
    embeddings = model.preprocess_inputs(inputs)  
    ## returns a list of length embeddings.shape[0] (batch size)
    output = model.generate(
        embeddings = embeddings,
        max_steps = 4,
        temperature = 0.1,
        top_k = 0,
    ) 

    print('output',output[0]) 
    
    print('answer1',answer1)
    print('answer2',answer2)

print("accuracy", c/100)
    
