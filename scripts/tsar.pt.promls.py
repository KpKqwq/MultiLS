#%%
import openai
import os
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
output0_texts=open("lspg_output/output.0.1.pt.txt",encoding="utf-8").readlines()
import time
with open("results/tsar.results.pt.promls","w+",encoding="utf-8") as f1:
    for line in output0_texts:
        sentence=line.strip().split("\t")[0].strip()
        complex_word=line.strip().split("\t")[1].strip()
        substitutions_lst=line.strip().split("\t")[2:2+10]
        for i in range(len(substitutions_lst)):
            #substitutions_lst[i]=substitutions_lst[i].split("(")[0].strip()
            substitutions_lst[i]=substitutions_lst[i]
            
        substitutions=",".join(substitutions_lst)
        
        substitutions_lst=substitutions.split(",")
        substitutions=give_one_hundred(substitutions_lst)
        
        prefix="Por favor, responda à pergunta de acordo com o contexto e responda aos candidatos. Os candidatos são gerados a partir de outro modelo pré-treinado. Cada candidato a resposta está associado a uma pontuação de confiança dentro de um colchete. Observe que a resposta pode não estar incluída nos candidatos."+"\n\n"

        line_prefix="===\n"
        
        context1=line_prefix+"Contexto: "+"esse mecanismo é o equivalente geológico de um cobertor numa noite fria que aquece a atmosfera da terra retendo radiação do sol que de outro modo se dissiparia no espaço"+"\n"

        question1=line_prefix+"Pergunta: Quais palavras podem simplificar melhor a palavra complexa '"+"retendo"+"' ?"+"\n"
        candidates1=line_prefix+"Candidatos: "+"detendo(73.36),retenendo(71.56),capturando(68.17),prendendo(55.76),atraindo(40.86),captando(38.83),trazendo(29.12),agarrando(26.64),absorvendo(25.51),retiver(22.57)"+"\n"
        answer1=line_prefix+"Resposta: "+"guardando  segurando  conservando  mantendo  detendo  absorvendo  possuindo  contendo  trazendo  prendendo."+"\n"
        example1=context1+question1+candidates1+answer1+"\n"

        
        

        context3=line_prefix+"Contexto: "+"bacci pretende oferecer recompensa a chamada delação premiada a bandidos que colaborarem com as investigações para desarticular as grandes quadrilhas"+"\n"
        question3=line_prefix+"Pergunta: Quais palavras podem simplificar melhor a palavra complexa '"+"desarticular"+"'?"+"\n"
        candidates3=line_prefix+"Candidatos: "+"desfazer(-7.36),quebrar(-9.19),derrubar(-9.26),destruir(-9.5),esmagar(-9.58),dispersar(-9.59),despedaçar(-9.69),romper(-10.03),separar(-10.35),desencadear(-10.48)"+"\n"
        answer3=line_prefix+"Resposta: "+"desmontar  separar  desfazer  desencaixar  exarticular  luxate"+"\n"
        example3=context3+question3+candidates3+answer3+"\n"




        context2=line_prefix+"Contexto: "+"naquele país a ave é considerada uma praga"+"\n"
        question2=line_prefix+"Pergunta: Quais palavras podem simplificar melhor a palavra complexa '"+"praga"+"' ?"+"\n"
        candidates1="Candidates: "+"use(-7.35),launch(-7.6),install(-7.73),dispatch(-7.83),move(-8.02),unleash(-8.32),send(-8.41),station(-8.5),field(-8.72),implement(-8.73)"+"\n"
        candidates2=line_prefix+"Candidatos: "+"peste(83.66),doença(61.44),plaga(49.67),maldição(35.29),infecção(34.8),desgraça(33.17),infestação(22.71),aflição(17.48),pestilência(16.67),pestana(16.34)"+"\n"
        answer2=line_prefix+"Resposta: "+"peste  epidemia  maldição  doença  tragédia  desgraça  infestação."+"\n"
        example2=context2+question2+candidates2+answer2+"\n"



        context=line_prefix+"Contexto: "+sentence+"\n"
        question=line_prefix+"Pergunta: Quais palavras podem simplificar melhor a palavra complexa '"+complex_word+"' ?"+"\n"
        candidates=line_prefix+"Candidatos: "+substitutions+"\n"
        answer=line_prefix+"Resposta:"

        prompt=prefix+example2+example1+context+question+candidates+answer
        print(prompt)
        
        
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