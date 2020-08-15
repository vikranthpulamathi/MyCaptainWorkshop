from operator import itemgetter

def most_frequent(str1):
    d = {}
    for n in str1:
        if n in d:
            d[n] += 1
        else:
            d[n] =1
    di = dict(sorted(d.items(), key = itemgetter(1), reverse = True))
    return di

str2 = str(input('Enter a string: '))
print(most_frequent(str2))
