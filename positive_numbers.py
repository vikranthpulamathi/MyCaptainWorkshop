def get_positives(n):
    return [num for num in n if num > 0]

# Use a different variable name to avoid shadowing the built-in input()
input = input("Enter a list of numbers separated by spaces: ")
input_list = list(map(int, input.split()))

positives = get_positives(input_list)
print("The positive numbers are:", positives)