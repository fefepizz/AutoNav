import csv

def count_non_zero_lines(file_path):
    count = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if any(cell != '0' for cell in row):
                count += 1
    return count

if __name__ == "__main__":
    file_path = 'line_points_cs2.csv'
    result = count_non_zero_lines(file_path)
    print(f"Number of lines without all zeros: {result}")