from monkeylearn import MonkeyLearn
 
ml = MonkeyLearn('35ab2ef38b1f683705009f64525d4398f27cbf6f')
text_list = ["I am very beautiful", "Stocks are evil."]

text_list2 = input("Write a phrase: ")
module_id = 'cl_b7qAkDMz'
res = ml.classifiers.classify(module_id, text_list2)
print (res.result)