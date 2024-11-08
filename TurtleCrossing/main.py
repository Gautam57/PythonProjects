import time
from turtle import Screen
from player import Player
from car_manager import CarManager
from scoreboard import Scoreboard

screen = Screen()
screen.setup(width=600, height=600)
screen.tracer(0)

player = Player()
screen.listen()
screen.onkey(player.move_player, 'Up')
scoreboard = Scoreboard()
car = CarManager()


game_is_on = True
while game_is_on:
    time.sleep(0.1)
    screen.update()
    car.create_car()
    car.move()
    if player.ycor() >= 270:
        player.finish_level()
        scoreboard.update_level()
        car.level_up()

    for dash in car.cars:
        if player.distance(dash) < 20:
            game_is_on = False
            scoreboard.game_over()

screen.exitonclick()

