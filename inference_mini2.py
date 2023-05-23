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
train_df=pd.read_csv('out.csv')
val_df=pd.read_csv('out.csv')

path_lst=glob.glob("/local1/monajati/kaggle/*") 
print(path_lst[0])
print(path_lst[1])

'n01532829 n01558993'

c=0
for i in range(5):
    
    print("i",i)
    
    ind1 = random.randint(0,len(val_df)-1)
    ind1=888
    
    im_id1=val_df['im_id'][ind1]
    
    paths1=glob.glob("/home/monajati/main/metaVL/magma/mini/ImageNet-Mini/images/"+im_id1+"/*")
    
    
    path1_1=paths1[0]
    
    answer1=val_df['label'][ind1].replace("_", " " )
    answer1='dax'
    
    path1_2=paths1[1]
    
    
    
    
    
    ind2 = ind1+1
    
    im_id2=val_df['im_id'][ind2]
    
    paths2=glob.glob("/home/monajati/main/metaVL/magma/mini/ImageNet-Mini/images/"+im_id2+"/*")
    
    path2_1=paths2[0]
    answer2=val_df['label'][ind2].replace("_", " ")
    answer2='blicket'
    
    path2_2=paths2[1]
    
    prompt= 'Answer with ' + answer1 + ' or ' + answer2 + '.'
    
    inputs =['Please answer the question.' ,
             ImageInput(path1_1),
    'Question: what is this? Answer: ' + answer1 ,
             'Please answer the question.' ,
             ImageInput(path2_1),
    'Question: what is this? Answer: '+ answer2 ,
             'Please answer the question.' ,
             ImageInput(paths1[1]),
    'Question: what is this? Answer: ' 
    ]
    
    inputs =[ImageInput(path1_1),
    'This is a ' + answer1 + '.' ,
             ImageInput(path2_1),
    'This is a '+ answer2  + '.',
             ImageInput(path1_2),
    'This is a ' + answer1 + '.' ,
             ImageInput(path2_2),
    'This is a '+ answer2  + '.',
             ImageInput(paths1[1]),
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
    
