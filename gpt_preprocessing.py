from openai import OpenAI
client = OpenAI()


response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with a legal case document that you have to preprocess. In particular, in the middle of many sentences there is a new line, so you have to keep the whole sentence in only one line. You have to preprocess the document WITHOUT changing any word."
    },
    {
      "role": "user",
      "content": open("/home/edo/PycharmProjects/coliee24/Dataset/Train_Evidence/094835.txt").read()
    }
  ],
  temperature=0.0,
  seed=62
)

print(response.choices[0].message.content)
print(response.system_fingerprint)
pass
