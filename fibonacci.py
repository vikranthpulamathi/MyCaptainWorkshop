def fibonacci(n):
  series = []
  a, b = 0, 1
  for _ in range(n):
    series.append(a)
    a, b = b, a+b
  return series
  
num = int(input("Enter the length of terms: " ))
fib_series = fibonacci(num)
print(fib_series)