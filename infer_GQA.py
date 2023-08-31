from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image
import torch
import json

from transformers import GPT2TokenizerFast, AutoTokenizer

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/meta_ori2/global_step29100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
model.eval()


val_df=pd.read_csv('val_gqa_5000.csv')
train_df=pd.read_csv('train_gqa.csv')


output_lst=[]
answer_lst=[]
question_lst=[]
ids=[]

c=0
for i in range(len(val_df)):
    
    print("i",i)

    
    
    ind=i
    
    
    im_id=val_df['image_id'][ind]
    
    image_path='/local1/monajati/magma/GQA2/images/'+str(im_id)+'.jpg'
    
    
    question=val_df['question'][ind]
    answer=val_df['answer'][ind]
    
    answer_lst.append(answer)
    question_lst.append(question)
    ids.append(str(im_id))
    
    
    n1 = random.randint(0,len(train_df))
    id1=train_df['image_id'][n1]
    
    image_1='/local1/monajati/magma/GQA2/images/'+str(id1)+'.jpg'
    
    n2 = random.randint(0,len(train_df))
    id2=train_df['image_id'][n2]
    
    image_2='/local1/monajati/magma/GQA2/images/'+str(id2)+'.jpg'
    
    n3 = random.randint(0,len(train_df))
    id3=train_df['image_id'][n3]
    
    image_3='/local1/monajati/magma/GQA2/images/'+str(id3)+'.jpg'
    print('Question: '+question+ ' Answer:')

    inputs =['Please answer the question.',
    ImageInput(image_1),
    'Question: '+train_df['question'][n1]+ ' Answer: ' + train_df['answer'][n1],
    'Please answer the question.',
    ImageInput(image_2),
    'Question: '+train_df['question'][n2]+ ' Answer: ' + train_df['answer'][n2],
    'Please answer the question.',
    ImageInput(image_3),
    'Question: '+train_df['question'][n3]+ ' Answer: ' + train_df['answer'][n3],
    'Please answer the question.',
    ImageInput(image_path),
    'Question: '+question+ ' Answer:'
    ]
    
    
    '''
    inputs =['Please answer the question.',
    ImageInput(image_path),
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
    
    print('answer',answer)

data = {'image_id':ids,
        'question':question_lst,
       'output':output_lst,
        'ans': answer_lst}


with open("outputs/GQA/GQA_zero_meta_adap.json", "w") as outfile:
    json.dump(data, outfile)
    
print("accuracy", c/100)
    
