from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image
import torch


from transformers import GPT2TokenizerFast, AutoTokenizer


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/meta_ori2/global_step29100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
model.eval()

val_df=pd.read_csv('val_ok_5000.csv')
train_df=pd.read_csv('train_ok.csv')

import json

output_lst=[]
answer_lst=[]
question_lst=[]
ids=[]

c=0
for i in range(len(val_df)):
    
    print("i",i)
    
    ind=i
    
    im_id=val_df['image_id'][ind]
    
    im_id=str(im_id)
    if len(im_id)!=6:
        for i in range(6-len(im_id)):
            im_id='0'+im_id
        
        print("im_id",im_id)
    
    question=val_df['question'][ind]

    answer=val_df['answer'][ind]
    
    
    n1 = random.randint(0,len(train_df['question'])-1)
    id1=train_df['image_id'][n1]
    id1=str(id1)
    
    if len(id1)!=6:
        for i in range(6-len(id1)):
            id1='0'+id1
        
    n2 = random.randint(0,len(train_df['question'])-1)
    id2=train_df['image_id'][n2]
    id2=str(id2)
    
    if len(id2)!=6:
        for i in range(6-len(id2)):
            id2='0'+id2
        
    n3 = random.randint(0,len(train_df['question'])-1)
    id3=train_df['image_id'][n3]
    id3=str(id3)
    
    if len(id3)!=6:
        for i in range(6-len(str(id3))):
            id3='0'+id3
        
    
    
    url1='http://images.cocodataset.org/train2017/000000'+im_id+'.jpg'
    url2='http://images.cocodataset.org/val2017/000000'+im_id+'.jpg'

    try:
        im=Image.open(requests.get(url1, stream=True).raw)
        true_url=url1
    except:
        im=Image.open(requests.get(url2, stream=True).raw)
        true_url=url2

    print("true_url",true_url)

    inputs =['Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000'+id1+'.jpg'),
    'Question: '+train_df['question'][n1]+ ' Answer: ' + str(train_df['answer'][n1]),
    'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000'+id2+'.jpg'),
    'Question: '+train_df['question'][n2]+ ' Answer: ' + str(train_df['answer'][n2]),
    'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000'+id3+'.jpg'),
    'Question: '+train_df['question'][n3]+ ' Answer: ' + str(train_df['answer'][n3]),
    'Please answer the question.',ImageInput(true_url),
    'Question: '+question+ ' Answer:']

    '''
    inputs =[
    'Please answer the question.',
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
    
    answer_lst.append(answer)
    question_lst.append(question)
    ids.append(str(im_id))
    output_lst.append(output[0])

    print('output',output[0]) 
    
    print('answer',answer)

#print("accuracy", c/100)

data = {'image_id':ids,
        'question':question_lst,
       'output':output_lst,
        'ans': answer_lst}


with open("outputs/OKVQA/OKVQA_zero_meta1.json", "w") as outfile:
    json.dump(data, outfile)
    
print("accuracy", c/100)
    
