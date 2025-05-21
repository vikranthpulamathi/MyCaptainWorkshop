def perform_set_operations(set1, set2):
    print("Set 1:", set1)
    print("Set 2:", set2)
    print("\nSet Operations:")
    print("Union:", set1.union(set2))
    print("Intersection:", set1.intersection(set2))
    print("Difference (Set1 - Set2):", set1.difference(set2))
    print("Difference (Set2 - Set1):", set2.difference(set1))
    print("Symmetric Difference:", set1.symmetric_difference(set2))

set1_input = input("Enter elements of Set 1 separated by spaces: ")
set2_input = input("Enter elements of Set 2 separated by spaces: ")

set1 = set(map(int, set1_input.split()))
set2 = set(map(int, set2_input.split()))

perform_set_operations(set1, set2)
