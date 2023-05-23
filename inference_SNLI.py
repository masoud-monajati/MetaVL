from magma import Magma
from magma.image_input import ImageInput
import pandas as pd
import random
import requests
from PIL import Image

import json

model = Magma.from_checkpoint(config_path = "configs/MAGMA_gpt_med.yml",checkpoint_path = "/local1/monajati/magma/checkpoints/meta_ori2/global_step29100/mp_rank_00_model_states.pt", device = 'cuda:0')

model=model.float()
#"../../check/global_step19945/mp_rank_00_model_states.pt"

#val_df=pd.read_csv('val_gqa.csv')
#train_df=pd.read_csv('train_gqa.csv')

with open('/local1/monajati/magma/SLNI/data/snli_ve_dev.jsonl', 'r') as json_file:
    json_list_val = list(json_file)

with open('/local1/monajati/magma/SLNI/data/snli_ve_train.jsonl', 'r') as json_file:
    json_list_train = list(json_file)

c=0
for i in range(100):
    
    print("i",i)

    
    ind = random.randint(0,len(json_list_val))
    
    result = json.loads(json_list_val[ind])
    
    im_id=result['Flickr30K_ID']
    
    image_path='/local1/monajati/magma/SLNI/data/data/images/'+str(im_id)+'.jpg'
    
    text= result['sentence1'] + result['sentence2'] + ' Answer with entailment or neutral or contradiction. This is '
    
    if result['gold_label']=='entailment':
        result['gold_label'] = 'an ' + result['gold_label']
    else:
        result['gold_label'] = 'a ' + result['gold_label']
        
    label=result['gold_label']
    
    #print(result)
    
    
    
    #text='sentence 1: '+result['sentence1']+ ' sentence 2: ' + result['sentence2']
    #print(text)
    
    #print(begh)
    
    #label=result['gold_label']
    
    
    id1 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id1])
    
    n1=result['Flickr30K_ID']
    
    image_path1='/local1/monajati/magma/SLNI/data/data/images/'+str(n1)+'.jpg'
    if result['gold_label']=='entailment':
        result['gold_label'] = 'an ' + result['gold_label']
    else:
        result['gold_label'] = 'a ' + result['gold_label']
    
    text1= result['sentence1'] + result['sentence2'] + ' Answer with entailment or neutral or contradiction. This is ' + result['gold_label']
    
    
    id2 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id2])

    n2=result['Flickr30K_ID']
    
    image_path2='/local1/monajati/magma/SLNI/data/data/images/'+str(n2)+'.jpg'
    
    if result['gold_label']=='entailment':
        result['gold_label'] = 'an ' + result['gold_label']
    else:
        result['gold_label'] = 'a ' + result['gold_label']
    
    text2=result['sentence1'] + result['sentence2'] + ' Answer with entailment or neutral or contradiction. This is ' + result['gold_label']
    
    
    id3 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id3])
    
    n3=result['Flickr30K_ID']
    
    image_path3='/local1/monajati/magma/SLNI/data/data/images/'+str(n3)+'.jpg'
    
    if result['gold_label']=='entailment':
        result['gold_label'] = 'an ' + result['gold_label']
    else:
        result['gold_label'] = 'a ' + result['gold_label']
    
    text3=result['sentence1'] + result['sentence2'] + ' Answer with entailment or neutral or contradiction. This is ' + result['gold_label']
    
    
    id4 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id4])
    
    n4=result['Flickr30K_ID']
    
    image_path4='/local1/monajati/magma/SLNI/data/data/images/'+str(n4)+'.jpg'
    
    text4=result['sentence1'] + result['sentence2'] + ' Question: entailment or neutral or contradiction? Answer: ' + result['gold_label']
    
    id5 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id5])
    
    n5=result['Flickr30K_ID']
    
    image_path5='/local1/monajati/magma/SLNI/data/data/images/'+str(n5)+'.jpg'
    
    text5=result['sentence1'] + result['sentence2'] + ' Question: entailment or neutral or contradiction? Answer: ' + result['gold_label']
    
    id6 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id6])
    
    n6=result['Flickr30K_ID']
    
    image_path6='/local1/monajati/magma/SLNI/data/data/images/'+str(n6)+'.jpg'
    
    text6=result['sentence1'] + ' entailment, neutral, or contradiction? ' + result['gold_label']
    
    id7 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id7])
    
    n7=result['Flickr30K_ID']
    
    image_path7='/local1/monajati/magma/SLNI/data/data/images/'+str(n7)+'.jpg'
    
    text7=result['sentence1'] + ' entailment, neutral, or contradiction? ' + result['gold_label']
    '''
    id8 = random.randint(0,len(json_list_train))
    
    result = json.loads(json_list_train[id8])
    
    n8=result['Flickr30K_ID']
    
    image_path8='../v-coco/SNLI-VE/data/data/images/'+str(n8)+'.jpg'
    
    text8=result['sentence1'] + ' entailment, neutral, or contradiction? ' + result['gold_label']
    
    print(text1)
    print(text2)
    print(text3)
    print(text4)
    print(text)
    
    
    inputs =[
    ## supports urls and path/to/image
    'Please answer with entailment or neutral or contradiction.',
    ImageInput(image_path1),
    text1,
    'Please answer with entailment or neutral or contradiction.',
    ImageInput(image_path2),
    text2,
    'Please answer with entailment or neutral or contradiction.',
    ImageInput(image_path3),
    text3,
    'Please answer with entailment or neutral or contradiction.',
    ImageInput(image_path),
    text
    ]
    '''
    inputs =[
        'Answer with entailment or neutral or contradiction.',
        ImageInput(image_path1),
             text1,
        'Answer with entailment or neutral or contradiction.',
             ImageInput(image_path2),
             text2,
        'Answer with entailment or neutral or contradiction.',
             ImageInput(image_path3),
             text3,
        'Answer with entailment or neutral or contradiction.',
             ImageInput(image_path),
             text
            ]
    
    
    '''
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
    ImageInput('http://images.cocodataset.org/train2017/000000'+id1+'.jpg'),
    'Answer the question: '+'Q: '+train_df['question'][n1]+ ' A: ' + train_df['answer'][n1],
    ImageInput('http://images.cocodataset.org/train2017/000000'+id2+'.jpg'),
    'Answer the question: '+'Q: '+train_df['question'][n2]+ ' A: ' + train_df['answer'][n2],
    ImageInput('http://images.cocodataset.org/train2017/000000'+id3+'.jpg'),
    'Answer the question: '+'Q: '+train_df['question'][n3]+ ' A: ' + train_df['answer'][n3],
    ImageInput(true_url),
    'Answer the question: '+'Q: '+question+ ' A:'
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
    ImageInput(image_path),
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
    print('answer',label)
    
    if label==output[0][-len(label):]:
        c+=1
        print("number of correct answer",c)

print("accuracy", c/100)