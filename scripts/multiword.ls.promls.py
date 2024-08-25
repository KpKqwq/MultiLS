# file reader
import os
import time
import json
import openai
from openai import OpenAI
client = OpenAI(
    api_key="",
    base_url=""
)
lspg_file_reader=[json.loads(line) for line in open("lspg_output/mwls.multibeam.json")]

promls_file_reader=lspg_file_reader



def prepare_example1():
    context1=line_prefix+"Context: "+"That prompted the military to deploy its largest warship, the BRP Gregorio del Pilar, which was recently acquired from the United States."+"\n"
    question1=line_prefix+"Question: which words or phrases can best simplify the complex word '"+"deploy"+"'?"+"\n"
    
    candidates1=line_prefix+"Candidates: "+"dispatch(-6.42),use(-6.49),send(-6.54),move(-6.64),field(-7.02),employ(-7.54),launch(-7.58),place(-7.78),mobilize(-7.94),implement(-7.96)"+"\n"
    answer1=line_prefix+"Answer:"+"send  post  use  position  employ  extend  launch  organize  release  station  send out  let loose."+"\n" 
    example1=context1+question1+candidates1+answer1+"\n"
    return example1

def prepare_example2():
    context2=line_prefix+"Context: "+"A Spanish government source, however, later said that banks able to cover by themselves losses on their toxic property assets will not be forced to remove them from their books while it will be compulsory for those receiving public help."+"\n"
    question2=line_prefix+"Question: which words or phrases can best simplify the complex word '"+"compulsory"+"'?"+"\n"
    candidates2=line_prefix+"Candidates: "+"mandatory(-1.62),obligatory(-2.9),required(-6.13),obliged(-6.97),binding(-7.16),mandated(-7.51),forced(-7.61),necessary(-8.3),imperative(-8.95),obligation(-9.17)"+"\n"
    answer2=line_prefix+"Answer: "+"mandatory  required  essential  forced  important  necessary  obligatory  unavoidable."+"\n"
    example2=context2+question2+candidates2+answer2+"\n"    
    return example2

write_file=open("results/mwls.chatgpt.json.promls","w+")
from tqdm import tqdm
for i in tqdm(range(len(lspg_file_reader))):

    sentence=(lspg_file_reader[i]["prefix"].strip()+lspg_file_reader[i]["target_word"].strip()+lspg_file_reader[i]["suffix"].strip()).strip()
    complex_word=lspg_file_reader[i]["target_word"]

    prefix="Please answer the question according to the context and the answer candidates. The candidates are generated from other pretrained model. Each answer candidate is associated with a confidence score in parentheses. Note that the true answer may not be included in the candidates."+"\n\n"      
    substitutions=""
    for i_subs in range(len(lspg_file_reader[i]["outputs"][:10])):
        tmp_word=lspg_file_reader[i]["outputs"][i_subs]
        tmp_word_score=float(lspg_file_reader[i]["outputs_scores"][i_subs])
        tmp_word_score=round(tmp_word_score,2)
        tmp_prompt_str=f"{tmp_word}({tmp_word_score}),"

        substitutions+=tmp_prompt_str
    substitutions=substitutions[:-1]


    line_prefix="===\n"
    context=line_prefix+"Context: "+sentence+"\n"
    question=line_prefix+"Question: which words or phrases can best simplify the complex word '"+complex_word+"'?"+"\n"
    candidates=line_prefix+"Candidates: "+substitutions+"\n"
    answer=line_prefix+"Answer:"

    example1=prepare_example1()
    example2=prepare_example2()

    prompt=prefix+example1+example2+context+question+candidates+answer
    if(i==0):
        print(prompt)
    try:
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            stream=False,
            temperature=0,
            max_tokens=100,
                
        )
    except:
        time.sleep(20)
        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=prompt,
            stream=False,
            temperature=0,
            max_tokens=100,               
        )        

    

    import time
    time.sleep(1)
    output_str=response.choices[0].text
    output_str=output_str.strip().lower()
    if(output_str.endswith('.')):
        output_str=output_str[:-1]
        output_str=output_str.strip()
    llm_subs=output_str.split("  ")

    promls_file_reader[i]["outputs"]=llm_subs
    write_str=json.dumps(promls_file_reader[i])
    write_file.write(write_str.strip()+"\n")
    write_file.flush()


write_file.close()
