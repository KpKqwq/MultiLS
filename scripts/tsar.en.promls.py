
def give_one_hundred(line):
    number=[]
    words=[]
    rs=[]
    word_lsts=line
    for word1 in word_lsts:
        index1=word1.find("(")
        index2=word1.find(")")
        try:
            number.append(float(word1[index1+1:index2].strip()))
        except:
            import pdb
            pdb.set_trace()
        words.append(word1[:index1].strip())
    
    max1=max(number)+1
    min1=min(number)-1
    for i in range(len(number)):
        number[i]=round(100*(number[i]-min1)/(max1-min1),2)
    for i in range(len(number)):
        tmp=words[i]+"("+str(number[i])+")"
        rs.append(tmp)
    
    real_rs=",".join(rs)
    return real_rs



import openai
import pickle
from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url=""
)


results_list=[]
output0_texts=open("lspg_output/output.0.1.en.txt",encoding="utf-8").readlines()
output0_texts=output0_texts[:100]
import time
from tqdm import tqdm
with open("results/tsar.results.en.promls","w+",encoding="utf-8") as f1:
    for i,line in tqdm(enumerate(output0_texts), total=len(output0_texts)):
        sentence=line.strip().split("\t")[0].strip()
        complex_word=line.strip().split("\t")[1].strip()
        #substitutions=",".join(line.strip().split("\t")[2:2+10]).strip()
        #substitutions=",".join(line.strip().split("\t")[-1].strip().split(",")[:10])
        
        substitutions_lst=line.strip().split("\t")[-1].strip().split(",")[:10]
        for i in range(len(substitutions_lst)):
            #substitutions_lst[i]=substitutions_lst[i].split("(")[0].strip()
            substitutions_lst[i]=substitutions_lst[i]
        substitutions=",".join(substitutions_lst)
        
        #substitutions_lst=substitutions.split(",")
        #substitutions=give_one_hundred(substitutions_lst)
        
        prefix="Please answer the question according to the context and the answer candidates. The candidates are generated from another pretrained model. Each answer candidate is associated with a confidence score in parentheses. Note that the answer may not be included in the candidates."+"\n\n"                
        
        #prefix="Please answer the question according to the context."+"\n\n"
        line_prefix="===\n"
        
        context1=line_prefix+"Context: "+"That prompted the military to deploy its largest warship, the BRP Gregorio del Pilar, which was recently acquired from the United States."+"\n"
        question1=line_prefix+"Question: which words can best simplify the complex word '"+"deploy"+"'?"+"\n"
        candidates1=line_prefix+"Candidates: "+"dispatch(-6.42),use(-6.49),send(-6.54),move(-6.64),field(-7.02),employ(-7.54),launch(-7.58),place(-7.78),mobilize(-7.94),implement(-7.96)"+"\n"
        answer1=line_prefix+"Answer:"+"send  post  use  position  employ  extend  launch  organize  release  situation  station  redistribute."+"\n"
        example1=context1+question1+candidates1+answer1+"\n"

             
        
        context2=line_prefix+"Context: "+"A Spanish government source, however, later said that banks able to cover by themselves losses on their toxic property assets will not be forced to remove them from their books while it will be compulsory for those receiving public help."+"\n"
        question2=line_prefix+"Question: which words can best simplify the complex word '"+"compulsory"+"'?"+"\n"
        candidates2=line_prefix+"Candidates: "+"mandatory(-1.62),obligatory(-2.9),required(-6.13),obliged(-6.97),binding(-7.16),mandated(-7.51),forced(-7.61),necessary(-8.3),imperative(-8.95),obligation(-9.17)"+"\n"                               
        answer2=line_prefix+"Answer: "+"mandatory  required  essential  forced  important  necessary  obligatory  unavoidable."+"\n"
        example2=context2+question2+candidates2+answer2+"\n"

        context3=line_prefix+"Context: "+"Rajoy's conservative government had instilled markets with a brief dose of confidence by stepping into Bankia, performing a U-turn on its refusal to spend public money to rescue banks."+"\n"
        question3=line_prefix+"Question: which words can best simplify the complex word '"+"instilled"+"'?"+"\n"
        candidates3=line_prefix+"Candidates: "+"infused(-4.53),injected(-5.4),inspired(-5.85),inculcated(-6.02),imbued(-6.73),filled(-6.81),provided(-6.98),implanted(-7.01),showered(-7.22),impressed(-7.77)"+"\n"
        answer3=line_prefix+"Answer: "+"infused  introduced  filled  impressed  ingrained  inspired  promoted  anewed  created  fixed  implanted."+"\n"
        example3=context3+question3+candidates3+answer3+"\n"


        context=line_prefix+"Context: "+sentence+"\n"
        question=line_prefix+"Question: which words can best simplify the complex word '"+complex_word+"'?"+"\n"
        candidates=line_prefix+"Candidates: "+substitutions+"\n"
        answer=line_prefix+"Answer:"

        prompt=prefix+example1+example2+context+question+candidates+answer

        
        
        try:
          completion = client.completions.create(
              model="gpt-3.5-turbo-instruct",
              prompt=prompt,
              stream=False,
              temperature=0,
              max_tokens=100
          )
        except:
          time.sleep(60)
          completion =  client.completions.create(
              model="gpt-3.5-turbo-instruct",
              prompt=prompt,
              stream=False,
              temperature=0,
              max_tokens=100
          )
          print("still wrong!!!")
        time.sleep(0.1)

        dict1={}
        dict1["content"]=completion.choices[0].text
        time.sleep(1)
        #import pdb;pdb.set_trace()
        import time
        time.sleep(1)
        output_str=completion.choices[0].text
        #output_str=response.choices[0].message.content
        output_str=output_str.strip().lower()
        if(output_str.endswith('.')):
            output_str=output_str[:-1]
            output_str=output_str.strip()
        #print(output_str)
        llm_subs=output_str.split("  ")
        llm_subs_lines="\t".join(llm_subs)
        write_labels_line=sentence+"\t"+complex_word+"\t"+llm_subs_lines
        f1.write(write_labels_line.strip()+"\n")
        f1.flush()
f1.close()



    
    
    














