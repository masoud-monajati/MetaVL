from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image
import torch

import nltk
#nltk.download('punkt')

from nltk import word_tokenize

from transformers import GPT2TokenizerFast, AutoTokenizer

#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
#tokenizer = AutoTokenizer.from_pretrained("gpt-medium", use_fast=False)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/meta_ori2/global_step29100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
#"../../check/global_step19945/mp_rank_00_model_states.pt"

val_df=pd.read_csv('val.csv')
train_df=pd.read_csv('train.csv')

dic_type={"What is":[89680 ,271771 , 94156, 415521], "What are":[48245 ,177703, 212764, 145054], "What object":[104485, 328000, 395532, 284537],
          "What kind":[104485, 328000, 395532, 284537], "What gender":[104485, 328000, 395532, 284537],
          "What game":[104485, 328000, 395532, 284537],"What color":[175313, 139803 , 146399, 200933], 
          "What time": [435125, 181382, 14732, 4213], "Is": [295400, 231018, 98152, 335], 
          "Are": [418145, 43286, 413630, 241215], "How many": [323232 , 41148, 345894, 442278],
         "Where": [208699, 257063, 268891, 240414], "Which": [413347, 172511, 406256, 26215]}

dic_type2={"What color":[338000, 411246, 5403, 67443, 259989, 305317, 393720, 110101, 319463, 135038, 169915, 240236, 149607, 51639, 8940, 347986, 439633, 286565, 335069, 382652, 134623, 152325, 139808, 159557, 90625, 96964, 46415, 260151, 245494, 407017, 21514, 172453, 342108, 251305, 337570, 312140, 116775, 42271, 406222, 180521, 306473, 227217, 17450, 266604, 11591, 311619, 243006, 64849, 322174, 284057, 144984, 398541, 121404, 293430, 120733, 96649, 263227, 423854, 39490, 204522, 42614, 122317, 178301, 316705, 240502, 376128, 35558, 182319, 162895, 19907, 45451, 349959, 285171, 252490, 224121, 174833, 96375, 410296, 64316, 16280, 21813, 345145, 251147, 430605, 42545, 200913, 396249, 55105, 266919, 420582, 144801, 193438, 269833, 419259, 79089, 435583, 6810, 95008, 219742]}

c=0
for i in range(100):
    
    print("i",i)
    
    ind = random.randint(0,len(val_df))
    
    im_id=val_df['image_id'][ind]
    
    im_id=str(im_id)
    if len(im_id)!=6:
        for i in range(6-len(im_id)):
            im_id='0'+im_id
        
        #print("im_id",im_id)
    
    question=val_df['question'][ind]
    
    print("question",question)
    
    words = word_tokenize(question)
    
    #print(begh)
    answer=val_df['answer'][ind]
    
    if words[0]+" "+ words[1] != "What color":
        n1 = random.randint(0,len(train_df))
        id1=train_df['image_id'][n1]
        id1=str(id1)

        if len(id1)!=6:
            for i in range(6-len(id1)):
                id1='0'+id1

            #print("id1",id1)
        '''    
        t_question=train_df['question'][n1]
        words = word_tokenize(t_question)

        if words[0]+ " " + words[1]=="How many":
            print("t_question",t_question)
            print("words[0]",words[0]+ " " + words[1])
            print("answer",train_df['answer'][n1])
            print("n1",n1)
            c+=1
        continue
        if c == 5:
            print(begh)
        '''

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
    
    else:
        qtype = "What color"
        print("What color type")
        lst_q=dic_type2[qtype]
        n1 = random.choice(lst_q)
        #n1= dic_type[qtype][0]
        id1=train_df['image_id'][n1]
        id1=str(id1)

        if len(id1)!=6:
            for i in range(6-len(id1)):
                id1='0'+id1

        #n2= dic_type[qtype][1]
        n2 = random.choice(lst_q)
        id2=train_df['image_id'][n2]
        id2=str(id2)

        if len(id2)!=6:
            for i in range(6-len(id2)):
                id2='0'+id2
                
        #n3= dic_type[qtype][2]
        n3 = random.choice(lst_q)
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


            #print("id3",id3)
    '''
    print("true_url",true_url)
    print("text1",'Q: '+train_df['question'][n1]+ ' A: ' + train_df['answer'][n1])
    print("text2",'Q: '+train_df['question'][n2]+ ' A: ' + train_df['answer'][n2])
    print("text3",'Q: '+train_df['question'][n3]+ ' A: ' + train_df['answer'][n3])
    print("final",'Q: '+question+ ' A:')
    
    
    
     
    
    print('Question: '+question+ ' Answer:')
    
    inputs =[
    ## supports urls and path/to/image
    'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000530047.jpg'),
    'Question: Is this a tablet computer on the desk? Answer: yes'
    'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000200103.jpg'),
    'Question: How many people are skiing? Answer: 2',
    'Please answer the question.',
    ImageInput('http://images.cocodataset.org/train2017/000000466217.jpg'),
    'Question: What is the dog on? Answer: sand',
    'Please answer the question.',
    ImageInput(true_url),
    'Question: '+question+ ' Answer:'
    ]
    
    print("true_url",true_url)
    
    '''
        
        
    inputs =[
    ## supports urls and path/to/image
    'Please answer the question.',
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
    #print(len(inputs))
    #print(inputs[11])
    #print(inputs[12])
    print('Question: '+question+ ' Answer:')
    
    '''
    
    
    inputs =[
    ## supports urls and path/to/image
    'Please answer the question.',
    ImageInput(true_url),
    'Question: '+question+ ' Answer:'
    ]
    
    '''
    '''
    inputs =[
    ## supports urls and path/to/image
    'Q: '+train_df['question'][n1]+ ' A: ' + train_df['answer'][n1],
    'Q: '+train_df['question'][n2]+ ' A: ' + train_df['answer'][n2],
    'Q: '+train_df['question'][n3]+ ' A: ' + train_df['answer'][n3],
    'Q: '+question+ ' A:'
    ]
    
    inputs =[
    ## supports urls and path/to/image
    'Please answer the question. '+'two men are watching TV. '+'question: '+'how many people are watching TV?'+ ' answer: ' + 'two',
    'Please answer the question. '+'a chef is cooking a cake. '+'question: '+'what is being cooked in the kitchen?'+ ' answer: ' + 'cake',
    'Please answer the question. '+'a boat on the lake. '+'question: what is on the lake?'+ ' answer: ' +'boat',
    'Please answer the question. '+'a pizza on the table. '+'question: What is on the table?'+ ' answer:'
    ]
    
    '''
    '''
    inputs = ['Please answer the question. two men are watching TV. question: how many people are watching TV? answer: 2Please answer the question. a chef is cooking a cake. question: what is being cooked in the kitchen? answer: cake please answer the question. a boat on the lake. question: what is on the lake? answer: boat please answer the question. a pizza on the table. question: What is on the table? answer:']
    
    
    
    inputs =[
    ## supports urls and path/to/image
    'Q: '+'does the text match to the image?'+ ' A: ' + 'yes',
    'Q: '+'does the text match to the image?'+ ' A: ' + 'yes',
    'Q: '+'does the text match to the image?'+ ' A: ' + 'yes',
    'Q: '+'does the text match to the image?'+ ' A: ' + 'yes',
    'Q: '+'does the text match to the image?'+ ' A: '
    ]
    
    
    
    url1='http://images.cocodataset.org/train2017/000000'+im_id+'.jpg'
    url2='http://images.cocodataset.org/val2017/000000'+im_id+'.jpg'

    try:
        im=Image.open(requests.get(url1, stream=True).raw)
        true_url=url1
    except:
        im=Image.open(requests.get(url2, stream=True).raw)
        true_url=url2

    print("true_url",true_url)
    
    
    inputs =[
    ## supports urls and path/to/image
    ImageInput(true_url),
    'Answer the question: '+'Q: '+question+ ' A:'
    ]
    '''
   
    embeddings = model.preprocess_inputs(inputs)  
    ## returns a list of length embeddings.shape[0] (batch size)
    output = model.generate(
        embeddings = embeddings,
        max_steps = 6,
        temperature = 0.1,
        top_k = 0,
    ) 

    print('output',output[0]) 
    
    print('answer',answer)
    
    if answer==output[0][-len(answer):]:
        c+=1
        print("number of correct answer",c)

print("accuracy", c/100)
    
