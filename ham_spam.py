import numpy as np
import re
import math

#below 4 constants need to be given for each training and testing data set

#total ham training files
ham = 1000
#total spam training files
spam = 997
#total ham testing files
testham = 400
#total spam testing files
testspam = 400

# initialization
total_words_in_ham = 0
total_words_in_spam = 0
wordDict = {}


# ****************************************************************************

# traning the data set

# ****************************************************************************


# should change the range size to actuall 1 to 1000 for ham
# reading ham class

for x in range(1, ham+1):
    x_str = str(x)
    zerostr = x_str.zfill(5)
    filename = "train-ham-"+zerostr+".txt"

    file1 = open('train/'+filename, 'r')

    Line_str = file1.read()
    word_list = re.split("[^a-zA-Z]", Line_str)
    word_list = [string for string in word_list if string != '']

    # update the total count of words in ham
    total_words_in_ham = total_words_in_ham + len(word_list)

    for word in word_list:
        if wordDict.__contains__(word.lower()):
            countlist = wordDict.get(word.lower())
            num = countlist[0]+1
            wordDict.update({word.lower(): [num, 0, 0, 0]})
        else:
            wordDict.update({word.lower(): [1, 0, 0, 0]})

# reading spam class

for x in range(1, spam+1):
    x_str = str(x)
    zerostr = x_str.zfill(5)

    filename = "train-spam-"+zerostr+".txt"
    file1 = open('train/'+filename, 'r')

    Line_str = file1.read()
    word_list = re.split("[^a-zA-Z]", Line_str)
    word_list = [string for string in word_list if string != '']

    # update the total count of words in ham
    total_words_in_spam = total_words_in_spam + len(word_list)

    for word in word_list:
        if wordDict.__contains__(word.lower()):
            countlist = wordDict.get(word.lower())
            hamnum = countlist[0]
            spamnum = countlist[1]+1
            wordDict.update({word.lower(): [hamnum, spamnum, 0, 0]})
        else:
            wordDict.update({word.lower(): [0, 1, 0, 0]})

vocab = len(wordDict)
for key in wordDict:
    countlist = wordDict.get(key)
    countlist[2] = (countlist[0] + 0.5)/(total_words_in_ham + math.sqrt(vocab))
    countlist[3] = (countlist[1] + 0.5) / \
        (total_words_in_spam + math.sqrt(vocab))
    wordDict.update({key: countlist})


# writng to a file Model.txt
counter = 1
file2 = open("model.txt", 'w')
for key in sorted(wordDict.keys()):
    countlist = wordDict.get(key)
    strwrd = str(counter) + "  " + key + "  " + \
        str(countlist[0]) + "  "+str(countlist[2]) + "  " + \
        str(countlist[1])+"  "+str(countlist[3])
    file2.write(strwrd)
    file2.write('\n')
    counter = counter+1

# ****************************************************************************

# testing

# ****************************************************************************

result_dict = {}

trueham = 0
falsespam = 0
truespam = 0
falseham = 0


# for Ham

for z in range(1, testham+1):
    z_str = str(z)
    zerostr = z_str.zfill(5)

    filename = "test-ham-"+zerostr+".txt"
    testfile = open('test/'+filename, 'r')

    Line_str = testfile.read()
    words_in_doc = re.split("[^a-zA-Z]", Line_str)
    words_in_doc = [string for string in words_in_doc if string != '']

    hamscore = math.log10(ham/(ham+spam))
    spamscore = math.log10(spam/(ham+spam))

    for word in words_in_doc:
        if (wordDict.__contains__(word.lower())):
            countlist = wordDict.get(word.lower())
            hamscore = hamscore + math.log10(countlist[2])
            spamscore = spamscore + math.log10(countlist[3])
        else:
            continue

    if (hamscore >= spamscore):
        trueham = trueham+1
        result_dict.update(
            {filename: ["ham", hamscore, spamscore, "ham", "right"]})
    else:
        falsespam = falsespam + 1
        result_dict.update(
            {filename: ["spam", hamscore, spamscore, "ham", "wrong"]})


# ****  for spam class

for z2 in range(1, testspam+1):
    z_str2 = str(z2)
    zerostr2 = z_str2.zfill(5)

    filename = "test-spam-"+zerostr2+".txt"
    testfile = open('test/'+filename, 'r', errors='ignore')

    Line_str = testfile.read()
    words_in_doc = re.split("[^a-zA-Z]", Line_str)
    words_in_doc = [string for string in words_in_doc if string != '']

    hamscore = math.log10(1000/1997)
    spamscore = math.log10(997/1997)
    for word in words_in_doc:
        if (wordDict.__contains__(word.lower())):
            countlist = wordDict.get(word.lower())
            hamscore = hamscore + math.log10(countlist[2])
            spamscore = spamscore + math.log10(countlist[3])
        else:
            continue
    
    if (hamscore <= spamscore):
        truespam = truespam+1
        result_dict.update(
            {filename: ["spam", hamscore, spamscore, "spam", "right"]})
    else:
        falseham = falseham + 1
        result_dict.update(
            {filename: ["ham", hamscore, spamscore, "spam", "wrong"]})


# writng to a file Result.txt
counter2 = 1
file3 = open("result.txt", 'w')
for key in sorted(result_dict.keys()):
    countlist2 = result_dict.get(key)
    strwrd2 = str(counter2) + "  " + key + "  " + \
        str(countlist2[0]) + "  "+str(countlist2[1]) + "  " + \
        str(countlist2[2])+"  "+str(countlist2[3])+"  " +str(countlist2[4])
    file3.write(strwrd2)
    file3.write('\n')
    counter2 = counter2+1


#ham class confusion matrix
tp_ham = trueham
fp_ham = falseham
fn_ham = falsespam
tn_ham = truespam

#spam class confusion matrix
tp_spam = truespam
fp_spam = falsespam
fn_spam = falseham
tn_spam = trueham

print("\nHAM class confusion matrix\n")

print("TP:",tp_ham,"    FP:",fp_ham)
print("FN:",fn_ham,"      TN:",tn_ham)

print("\nSPAM class confusion matrix\n")

print("TP:",tp_spam, "    FP:",fp_spam)
print("FN:",fn_spam, "     TN:",tn_spam)

print("\nHAM class evaluation:\n")

print("Accuracy:",str(round(((tp_ham+tn_ham)/(tp_ham+tn_ham+fn_ham+fp_ham))*100, 2)),"%")

precision = tp_ham/(tp_ham+fp_ham)
recall = tp_ham/(tp_ham+fn_ham)

print("Precision:", str(round(precision*100, 2)),"%")
print("Recall:", str(round(recall*100, 2)),"%")
print("F1-measure(beta=1)", str(round(((2*precision*recall)/(precision+recall))*100, 2)),"%")

print("\nSPAM class evaluation:\n")

print("Accuracy:",str(round(((tp_spam+tn_spam)/(tp_spam+tn_spam+fn_spam+fp_spam))*100, 2)),"%")

precision = tp_spam/(tp_spam+fp_spam)
recall = tp_spam/(tp_spam+fn_spam)

print("Precision:", str(round(precision*100, 2)),"%")
print("Recall:", str(round(recall*100, 2)),"%")
print("F1-measure(beta=1)", str(round(((2*precision*recall)/(precision+recall))*100, 2)),"%\n")