from sentence_transformers import SentenceTransformer
import pdb
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
sbert_model = None
def _get_sbert_model():
    global sbert_model
    if not sbert_model:
        sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return sbert_model
sbert =_get_sbert_model()


#results=pd.read_csv('outputs/VQA_zero.json')
import json
with open('outputs/VQA_med_few.json') as f:
    results = json.load(f)
    
with open('/home/monajati/main/metaVL/magma/CLIP/CLIP-ViL/CLIP-ViL-Pretrain/data/vqa/trainval_label2ans.json') as f:
    all_answers = json.load(f)

    
c=0
tot=-1
for i in range(len(results['image_id'])):
    tot+=1
    print('i',i)
    print('tot',tot)
    print('c',c)
    ids = results['image_id'][i]
    output = results['output'][i].lower()
    if len(output)==0:
        continue
    if output[-1]=='.':
        output = output[:-1]
    answer = results['ans'][i]
    if answer.isdigit():
        if output == answer:
            c+=1
            continue

    if 'yes' in output or 'no' in output:
        if answer in output:
            c+=1
            #print('begh')
    else:
        #can_answers = results['can_ans'][i]
        can_answers = all_answers

        features1 = sbert.encode(output)

        cos_sim=0
        for j in range(len(can_answers)):
            #print(can_answers[j])
            features2 = sbert.encode(can_answers[j])
            new_cos=cosine_similarity([list(features1)],[list(features2)])[0][0]
            if new_cos>cos_sim:
                cos_sim= new_cos
                new_answer=can_answers[j]

        #print("new_answer",new_answer)

        if answer==new_answer:
            c+=1
        else:
            print('output',output)
            #print('can_answers',can_answers)
            print('ans',answer)
            print('new_answer',new_answer)
print("number of correct answer",c)
print("accuracy", c/len(results['image_id']))
print("new_accuracy", c/tot)




