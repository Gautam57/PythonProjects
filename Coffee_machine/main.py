MENU = {
    "espresso": {
        "ingredients": {
            "water": 50,
            "coffee": 18,
            "milk": 0
        },
        "cost": 1.5,
    },
    "latte": {
        "ingredients": {
            "water": 200,
            "milk": 150,
            "coffee": 24,
        },
        "cost": 2.5,
    },
    "cappuccino": {
        "ingredients": {
            "water": 250,
            "milk": 100,
            "coffee": 24,
        },
        "cost": 3.0,
    }
}

resources = {
    "water": 300,
    "milk": 200,
    "coffee": 100,
}


# TODO: 4) Resource check for Making a drink
def resource_check(choice):
    if MENU[choice]["ingredients"]["water"] > resources['water']:
        print("Sorry there is not enough water .")
        return False
    elif MENU[choice]["ingredients"]["milk"] > resources['milk']:
        print("Sorry there is not enough milk .")
        return False
    elif MENU[choice]["ingredients"]["coffee"] > resources['coffee']:
        print("Sorry there is not enough coffee .")
        return False
    else:
        print("Sufficient Resources present for making the coffee")
        return True


def sufficient_money_check(choice, money):
    if MENU[choice]["cost"] > money:
        print("Sorry that's not enough money. Money refunded.")
        return False
    else:
        return True


# TODO: 1) Getting requests from user
Not_off = True
Money = 0
while Not_off:
    User_Request = input("What would you like ? (espresso/latte/cappuccino) ").lower()
    # TODO: 2) if off is entered in prompt end the code
    if User_Request == 'off':
        print("Machine Offed")
        Not_off = False

    sufficient_resource = True
    while sufficient_resource:
        # TODO: 3) If report entered in the prompt display the current resources
        if User_Request == 'report':
            print(f" Water: {resources['water']}ml \n Milk: {resources['milk']}ml \n "
                  f"Coffee: {resources['coffee']}g \n Money: ${Money} ")
            sufficient_resource = False
        if User_Request == 'espresso' or User_Request == 'latte' or User_Request == 'cappuccino':
            sufficient_resource = resource_check(User_Request)
            # TODO: 5) Receiving money and calculating it
            if sufficient_resource:
                quarters = int(input("how many quarters?: "))
                dimes = int(input("how many dimes?: "))
                nickles = int(input("how many nickels?: "))
                pennies = int(input("how many pennies?: "))
                user_given_amount = round(float((quarters*0.25)+(dimes*0.10)+(nickles*0.05)+(pennies*0.01)), 2)
                sufficient_resource = sufficient_money_check(User_Request, user_given_amount)
                if sufficient_resource:
                    Money += MENU[User_Request]["cost"]
                    resources["water"] -= MENU[User_Request]["ingredients"]["water"]
                    resources["milk"] -= MENU[User_Request]["ingredients"]["milk"]
                    resources["coffee"] -= MENU[User_Request]["ingredients"]["coffee"]
                    # TODO: 6) Returning the excess money
                    if user_given_amount > MENU[User_Request]["cost"]:
                        change = round(float(user_given_amount - MENU[User_Request]["cost"]), 2)
                        print(f"Here is ${change} dollars in change.")
                        print(f"Here is your {User_Request}. Enjoy!")
                        sufficient_resource = False

