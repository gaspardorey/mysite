import requests
import json

data = {
    'text_list': ["I am very happy.",
				  "I love McDonalds",
				  "I love McDonalds",
				  "I am a worker from New york."]
}

response = requests.post(
    "https://api.monkeylearn.com/v2/classifiers/cl_oJNMkt2V/classify/",
    data=json.dumps(data),
    headers={'Authorization': 'Token 35ab2ef38b1f683705009f64525d4398f27cbf6f',
            'Content-Type': 'application/json'})

print (json.loads(response.text))
