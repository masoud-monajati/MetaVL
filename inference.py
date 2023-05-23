from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image
import torch

from tqdm import tqdm
import json



#import nltk
#nltk.download('punkt')

#from nltk import word_tokenize

from transformers import GPT2TokenizerFast, AutoTokenizer

#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
#tokenizer = AutoTokenizer.from_pretrained("gpt-medium", use_fast=False)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/few_meta_50per/global_step3100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
model.eval()
#"../../check/global_step19945/mp_rank_00_model_states.pt"

val_df=pd.read_csv('val.csv')
train_df=pd.read_csv('train.csv')
val_df=pd.read_csv('val_vqa_5000.csv')
'''
import json
with open('../VQA/sample.json') as f:
    val_df = json.load(f)
'''
output_lst=[]
answer_lst=[]
question_lst=[]
ids=[]

c=0
for i in range(len(val_df)):
    
    print("i",i)

    
    #ind = random.randint(0,len(val_df))
    ind=i
    #ind = random.randint(0,len(val_df1)-1)
    #ind=i
    
    im_id=val_df['image_id'][ind]
    
    im_id=str(im_id)
    if len(im_id)!=6:
        for i in range(6-len(im_id)):
            im_id='0'+im_id
        
        #print("im_id",im_id)
    
    question=val_df['question'][ind]
    answer=val_df['answer'][ind]
    #can_answers=val_df['can_answers'][ind]
    
    
    n1 = random.randint(0,len(train_df))
    id1=train_df['image_id'][n1]
    id1=str(id1)
    
    if len(id1)!=6:
        for i in range(6-len(id1)):
            id1='0'+id1
        
        #print("id1",id1)
    
    n2 = random.randint(0,len(train_df))
    id2=train_df['image_id'][n2]
    id2=str(id2)
    
    if len(id2)!=6:
        for i in range(6-len(id2)):
            id2='0'+id2
        
        #print("id2",id2)
        
    n3 = random.randint(0,len(train_df))
    id3=train_df['image_id'][n3]
    id3=str(id3)
    
    if len(id3)!=6:
        for i in range(6-len(str(id3))):
            id3='0'+id3
        
        #print("id3",id3)
    
    
    url1='http://images.cocodataset.org/train2017/000000'+im_id+'.jpg'
    url2='http://images.cocodataset.org/val2017/000000'+im_id+'.jpg'

    try:
        im=Image.open(requests.get(url1, stream=True).raw)
        true_url=url1
    except:
        im=Image.open(requests.get(url2, stream=True).raw)
        true_url=url2
    #print("true_url",true_url)   
        
    
    inputs =['Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000'+id1+'.jpg'),
    'Question: '+train_df['question'][n1]+ ' Answer: ' + train_df['answer'][n1],
    'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000'+id2+'.jpg'),
    'Question: '+train_df['question'][n2]+ ' Answer: ' + train_df['answer'][n2],
             'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000'+id3+'.jpg'),
    'Question: '+train_df['question'][n3]+ ' Answer: ' + train_df['answer'][n3],
    'Please answer the question.',
    ImageInput(true_url),
    'Question: '+question+ ' Answer:'
    ]
    
    '''
    inputs =['Please answer the question.',
    ImageInput(true_url),
    'Question: '+question+ ' Answer:'
    ]
    
    
    '''
    
    
   
    embeddings = model.preprocess_inputs(inputs)  
    ## returns a list of length embeddings.shape[0] (batch size)
    output = model.generate(
        embeddings = embeddings,
        max_steps = 7,
        temperature = 0.1,
        top_k = 0,
    ) 

    print('output',output[0]) 
    
    output_lst.append(output[0])
    #can_answer_lst.append(can_answers)
    answer_lst.append(answer)
    ids.append(str(im_id))
    question_lst.append(question)
    
    
    print('answer',answer)
    
    
    
    
    if answer==output[0][-len(answer):]:
        c+=1
        #print("number of correct answer",c)

#print("accuracy", c/100)

data = {'image_id':ids,
        'question':question_lst,
       'output':output_lst,
        'ans': answer_lst}
'''
output_df = pd.DataFrame(data)

output_df.to_csv("outputs/VQA_zero.csv")
'''

with open("outputs/VQA_50/VQA_few_meta.json", "w") as outfile:
    json.dump(data, outfile)
    