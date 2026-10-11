from openai import OpenAI

client = OpenAI(
  base_url="http://192.168.1.5:1234/v1",
  api_key="1111"
)

messages=[{"role": "assistant", "content":  '''
Ты пощник. тебе надо помощь польователю решить задачу.
Когда пользователь задаст вопрос, то
с помощью функций узнай все необходимые данные.
не пеши \\n как переход на другую строку просто делай переход 
'''}]

text_str = '''
Пользовать может исполнять функции:
- прочитай_фаил(нозвание)
- напиши_в_фаил(нозвание, текст)

Например, если пользаватель запрашивает "прочитать фаил", то уточни у него нозвание файла и выведи в ответ только функцию в формате:
$read$название_фаила$read$
Например, если пользаватель запрашивает "напиши в фаил", то уточни у него нозвание файла и уточни текст или придумай сам для записи в фаил и выведи в ответ только функцию в формате:
$write$название_фаила№текст$write$
Например, если пользаватель запрашивает пока,выведи в ответ только функцию в формате:
$off$$off$
!!! Обязательно. Если в истории сообщений нет ответа на функцию, которую ты запросол, запроси повторно !!!
'''

messages.append({"role": "assistant", "content":  text_str})

user_text=''
                                     



f_user = 1

while True:
  if f_user==1:
      user_text = input("> ")
      
      if "конец" == user_text.lower():
        exit("конец")
     
      messages.append({"role": "user", "content": user_text})

  response =  client.chat.completions.create(
    model="Model",
    messages=messages
  )

  assistant_text = response.choices[0].message.content

  if "$off()$" in assistant_text:
    exit("конец")

  messages.append({"role": "assistant", "content": assistant_text})

  f_user=1
  if "$read$" in assistant_text:
    print(assistant_text)
    arg = assistant_text.split("$read$")[1]
    read_fail = open(arg, "r", encoding = "utf-8")
    otvet = read_fail.read()
    messages.append({"role": "user", "content": "Фаил текст(успешно): "+otvet})
    f_user=0
  elif "$write$" in assistant_text:
    print(assistant_text)
    neobrabot = assistant_text.split("$write$")[1]
    arg = neobrabot.split("№")[0]
    text = neobrabot.split("№")[1]
    read_fail = open(arg, "w", encoding = "utf-8")
    otvet = read_fail.write(text)

    messages.append({"role": "user", "content": "Записано в фаил(успешно): "+text})
    f_user=0
  elif "$off$" in assistant_text:
    exit()
  else:
    print(f"\033[36m{assistant_text}\033[0m")

