
import openai
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
#results_list=results_list
output0_texts=open("lspg_output/output.0.1.es.txt",encoding="utf-8").readlines()
import time

with open("results/tsar.results.es.promls","w+",encoding="utf-8") as f1:
    for line in output0_texts:
        sentence=line.strip().split("\t")[0].strip()
        complex_word=line.strip().split("\t")[1].strip()
        #substitutions=",".join(line.strip().split("\t")[2:2+10]).strip()
        substitutions_lst=line.strip().split("\t")[2:2+10]
        for i in range(len(substitutions_lst)):
            #substitutions_lst[i]=substitutions_lst[i].split("(")[0].strip()
            substitutions_lst[i]=substitutions_lst[i]
            
        substitutions=",".join(substitutions_lst)
        
        prefix="Por favor, responda a la pregunta de acuerdo con el contexto y las opciones de respuesta. Las opciones de respuesta se generan a partir de otro modelo preentrenado. Cada opción de respuesta está asociada con una puntuación de confianza entre corchetes. Ten en cuenta que la respuesta puede no estar incluida en los candidatos."+"\n\n"
        line_prefix=""
        context1=line_prefix+"Contexto: "+"Además de partidos de fútbol americano, el estadio ha sido utilizado para una gran variedad de eventos, entre los que se destacan varios partidos de la selección nacional de fútbol de los Estados Unidos, y fue el hogar del ahora difunto club de la MLS, el Tampa Bay Mutiny."+"\n"
        question1=line_prefix+"Pregunta: ¿qué palabras pueden simplificar mejor la compleja palabra \""+"difunto"+"\"?"+"\n"

        candidates1=line_prefix+"Candidatos: "+"fallecido(-3.54),desaparecido(-4.06),defunto(-5.49),extinto(-5.73),muerto(-6.91),finado(-7.88),agotado(-8.13),falecido(-8.61),pasado(-9.0),disuelto(-9.28)"+"\n"
        answer1=line_prefix+"Respuesta: "+"extinto  muerto  fallecido  finado  desaparecido  acabado  inactivo  inexistente."+"\n"
        example1=context1+question1+candidates1+answer1+"\n"
        #example1=context1+question1+answer1+"\n"


        context3=line_prefix+"Contexto: "+"bacci pretende oferecer recompensa a chamada delação premiada a bandidos que colaborarem com as investigações para desarticular as grandes quadrilhas"+"\n"
        question3=line_prefix+"Pregunta: ¿qué palabras pueden simplificar mejor la compleja palabra \""+"desarticular"+"\"?"+"\n"
        candidates3=line_prefix+"Candidatos: "+"desfazer(80.47),quebrar(44.73),derrubar(43.36),destruir(38.67),esmagar(37.11),dispersar(36.91),despedaçar(34.96),romper(28.32),separar(22.07),desencadear(19.53)"+"\n"
        answer3=line_prefix+"Respuesta: "+"desmontarseparar desfazer  desencaixar exarticular luxate"+"\n"
        example3=context3+question3+candidates3+answer3+"\n"


        context2=line_prefix+"Contexto: "+"Floreció en la época clásica y tenía una reputada escuela de filosofía."+"\n"
        question2=line_prefix+"Pregunta: ¿qué palabras pueden simplificar mejor la compleja palabra \""+"reputada"+"\"?"+"\n"
        candidates2=line_prefix+"Candidatos: "+"famosa(-3.18),prestigiosa(-3.92),renombrada(-4.05),reconocida(-4.33),conocida(-4.47),notable(-4.81),destacada(-4.94),célebre(-5.56),notoria(-5.71),importante(-5.97)"+"\n"    
        answer2=line_prefix+"Respuesta: "+"prestigiosa  famosa  reconocida  afamada  conocida  renombrada  respetada  prestigioso  acreditada  valorada."+"\n"
        example2=context2+question2+candidates2+answer2+"\n"

        context4=line_prefix+"Contexto: "+"El representativo chileno obtuvo una muy buena participación al conquistar los tres primeros lugares del citado certamen."+"\n"
        question4=line_prefix+"Pregunta: ¿qué palabras pueden simplificar mejor la compleja palabra \""+"conquistar"+"\"?"+"\n"
        candidates4=line_prefix+"Candidatos: "+"ganar(-6.25),conseguir(-7.73),capturar(-8.0),ocupar(-8.06),obtener(-8.08),lograr(-8.13),adquirir(-8.46),alcanzar(-8.51),tomar(-8.64),ganarse(-9.63)"+"\n"
        answer4=line_prefix+"Respuesta:"+"ganar  dominar  vencer  ocupar  tomar  obtener  lograr  colonizar  invadir  apoderarse."+"\n"
        example4=context4+question4+candidates4+answer4+"\n"



        context=line_prefix+"Contexto: "+sentence+"\n"
        question=line_prefix+"Pregunta: ¿qué palabras pueden simplificar mejor la compleja palabra \""+complex_word+"\"?"+"\n"
        candidates=line_prefix+"Candidatos: "+substitutions+"\n"
        answer=line_prefix+"Respuesta:"

        prompt=prefix+example1+example2+context+question+candidates+answer
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
