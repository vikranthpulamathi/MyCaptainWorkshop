def fibo(n):
    if n <= 1:
        return n
    else:
        return(fibo(n-1) + fibo(n-2))

n_terms = int(input("How many terms:" ))

if n_terms<=0:
    print("Check your number and run again.")
else:
    print("The Fibonacci Sequence is: ")
    for i in range(n_terms):
        print(fibo(i))