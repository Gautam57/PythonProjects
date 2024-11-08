import turtle
import pandas

# Create the graphics window
screen = turtle.Screen()

# Add the image as the background
image = "blank_states_img.gif"
screen.addshape(image)
turtle.shape(image)

# Read the state data from the CSV file
data = pandas.read_csv("50_states.csv")
state = data.state.to_list()
X = data.x.to_list()
Y = data.y.to_list()

# Lists to keep track of guessed and missed states
guessed_states = []
missed_states = []

# Counter for correct guesses
correct = 0

# Main game loop
while correct <= 50:
    # Prompt the user to guess a state
    answer_state = screen.textinput(title="Guess the state", prompt="What's another state's name? ").title()

    # Check if the user wants to exit the game
    if answer_state == 'Exit':
        # Check for missed states and save them to a CSV file
        for s in state:
            if s not in guessed_states:
                missed_states.append(s)
        missed_data = pandas.DataFrame(missed_states)
        missed_data.to_csv("missed states.csv")
        break

    # Check if the guessed state is correct
    for s in state:
        if s == answer_state:
            # Get the index of the matched state
            i = state.index(answer_state)

            # Create a turtle object and set its attributes
            t = turtle.Turtle()
            new_x = X[i]
            new_y = Y[i]
            t.hideturtle()
            t.penup()
            t.goto(new_x, new_y)
            t.write(s)

            # Update the counters and lists
            correct += 1
            guessed_states.append(answer_state)

    # Update the game window title to show progress
    screen.title(f"{correct}/50 U.S. States Game")

# Close the game window when clicked
screen.exitonclick()
