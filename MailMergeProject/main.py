# TODO: Create a letter using starting_letter.txt
# for each name in invited_names.txt
# Replace the [name] placeholder with the actual name.
# Save the letters in the folder "ReadyToSend".

with open('../Mail Merge Project Start/Input/Names/invited_names.txt', mode='r') as invited_names:
    invite_name = invited_names.readlines()

names = []
for i in invite_name:
    in_name = i.replace("\n", "")
    names.append(in_name)
print(names)
with open('../Mail Merge Project Start/Input/Letters/starting_letter.txt', mode='r') as Start_letter:
    start_letter = Start_letter.read()
    print(start_letter)
separated_letters = []
for n in names:
    separated_letter = start_letter.replace("[name]", n)
    separated_letters.append(separated_letter)

print(separated_letters)
for n in range(len(separated_letters)):
    with open(f'../Mail Merge Project Start/Output/ReadyToSend/letter_for_{names[n]}.txt', mode='w') as out:
        out.write(separated_letters[n])


