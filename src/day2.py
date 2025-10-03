
import csv

def main():
    with open('students.csv', mode='r') as file:
        reader = csv.DictReader(file)
        names_over_18 = [row['Name'] for row in reader if int(row['Age']) > 18]
    print(names_over_18)

if __name__ == "__main__":
    main()
