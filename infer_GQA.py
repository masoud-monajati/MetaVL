from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image
import torch
import json

'''
# Read the CSV file into a DataFrame
df = pd.read_csv('val_ok.csv')

# Get the number of rows in the DataFrame
n = len(df)

# Generate a list of 5000 random row indices
indices = random.sample(range(n), 5000)

# Select the 5000 rows using the indices
sampled_df = df.iloc[indices]

# Write the sampled DataFrame to a new CSV file
sampled_df.to_csv('val_ok_5000.csv', index=False)

print(begh)
'''

#import nltk
#nltk.download('punkt')

#from nltk import word_tokenize

from transformers import GPT2TokenizerFast, AutoTokenizer

#tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
#tokenizer = AutoTokenizer.from_pretrained("gpt-medium", use_fast=False)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/meta_ori2/global_step29100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
model.eval()

#model=model.float()
#"../../check/global_step19945/mp_rank_00_model_states.pt"

val_df=pd.read_csv('val_gqa_5000.csv')
train_df=pd.read_csv('train_gqa.csv')


output_lst=[]
answer_lst=[]
question_lst=[]
ids=[]

c=0
for i in range(len(val_df)):
    
    print("i",i)

    
    #ind = random.randint(0,len(val_df))
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
    
    '''
    inputs =['Please answer the question.',
    ImageInput(image_path),
    'Question: '+question+ ' Answer:'
    ]
    
    '''
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
    
    print('Please answer the question.',
    (image_1),
    'Question: '+train_df['question'][n1]+ ' Answer: ' + train_df['answer'][n1],
    'Please answer the question.',
    (image_2),
    'Question: '+train_df['question'][n2]+ ' Answer: ' + train_df['answer'][n2],
    'Please answer the question.',
    (image_3),
    'Question: '+train_df['question'][n3]+ ' Answer: ' + train_df['answer'][n3],
    'Please answer the question.',
    (image_path),
    'Question: '+question+ ' Answer:')
    
    '''
    inputs =['Please answer the question.',
    ImageInput(image_path),
    'Question: '+question+ ' Answer:'
    ]
    
    
    
    print("image_path",image_path)
    print('Please answer the question.', 
    image_1,
    'Question: '+train_df['question'][n1]+ ' Answer: ' + train_df['answer'][n1],
    'Please answer the question.',
    image_2,
    'Question: '+train_df['question'][n2]+ ' Answer: ' + train_df['answer'][n2],
    'Please answer the question.',
    image_3,
    'Question: '+train_df['question'][n3]+ ' Answer: ' + train_df['answer'][n3],
    'Please answer the question.',
    image_path,
    'Question: '+question+ ' Answer:')
    '''
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
'''
output_df = pd.DataFrame(data)

output_df.to_csv("outputs/VQA_zero.csv")
'''

with open("outputs/GQA/GQA_zero_meta_adap.json", "w") as outfile:
    json.dump(data, outfile)
    
print("accuracy", c/100)
    
