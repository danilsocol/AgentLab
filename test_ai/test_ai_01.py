from openai import OpenAI

client = OpenAI(
  base_url="http://192.168.1.5:1234/v1",
  api_key="1111"
)

messages=[{"role": "assistant", "content":  '''
Ты пощник. теюе надо помощь польователю решить задачу.
Когда пользователь задаст вопрос, то
с помощью функций узнай все необходимые данные, что бы ему было удобно по времени(успел за день) и не холодно
и дай ответ пользователю
'''}]

text_str = '''
Пользовать может исполнять функции:
- узнай_погоду(город)
- узнай_часовой_пояс(город)
- узнай_курс_валюты()

Например, ели пользователь запрашивает погоду, то уточни у него город и выведи в ответ только функцию в формате:
$fuction$узнай_погоду(Москва)$fuction$

!!! Обязательно. Если в истории сообщений нет ответа на функцию, которую ты запросол, запроси повторно !!!
'''

messages.append({"role": "assistant", "content":  text_str})

user_text=''
                                     
def get_water(city):
    return 'Дождливо, +15 градуслв'

def get_time_line(city):
    return '+3 часа от Москвы'

def get_value_rub(city):
    return '80 рублей за один доллар'



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
  print(f"\033[36m{assistant_text}\033[0m")

  f_user=1
  if "$fuction$" in assistant_text:
    function_str = assistant_text.split("$fuction$")[1]
    print(function_str)
    function_name=function_str.split('(')[0]
    function_args_str=function_str.split('(')[1][:-1]
    function_args=function_args_str.split(',')
    otvet=''
    if 'узнай_погоду' == function_name:
        otvet=get_water(function_args[0])
    if 'узнай_часовой_пояс' == function_name:
        otvet=get_time_line(function_args[0])
    if 'узнай_курс_валюты' == function_name:
        otvet=get_value_rub()
    print(otvet)
    messages.append({"role": "user", "content": otvet})
    f_user=0

