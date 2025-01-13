from typing import List
import itertools

def luhn_algorithm(card_number: str) -> bool:
    digits = [int(d) for d in card_number if d.isdigit()]
    checksum = 0
    for i, digit in enumerate(digits[::-1]):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0

def generate_combinations(pattern: str) -> List[str]:
    unknowns = [i for i, char in enumerate(pattern) if char == 'X']
    base_number = ''.join(char for char in pattern if char.isdigit() or char == 'X')
    
    valid_combinations = []
    for combo in itertools.product(range(10), repeat=len(unknowns)):
        card_number = list(base_number)
        for idx, digit in zip(unknowns, combo):
            card_number[idx] = str(digit)
        card_number_str = ''.join(card_number)
        if luhn_algorithm(card_number_str):
            valid_combinations.append(card_number_str)
    
    return valid_combinations

# Example usage
pattern = "5XX7 3790 7962 2XX6"
valid_combinations = generate_combinations(pattern)
for valid in valid_combinations:
    print(f"{valid[:4]} {valid[4:8]} {valid[8:12]} {valid[12:]}")   