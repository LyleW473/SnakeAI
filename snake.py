from pygame.draw import rect as pygame_draw_rect
from random import choice as random_choice
import numpy as np

class SnakeAI:
    
    def __init__(self, x, y):

        self.current_direction =  random_choice(("L", "R", "U", "D"))

        self.parts = [[x, y]]
        
        self.occupied_cells = set((x, y))

        self.move_timer = 0

        self.keys_released = True # Ensures that the player cannot hold onto a key

    def draw(self, surface, width, height):

        for i in range(len(self.parts)):

            colour = "RED" if i == 0 else "BLUE"
            pygame_draw_rect(
                            surface = surface,
                            color = colour, 
                            rect = (self.parts[i][0], self.parts[i][1], width, height),
                            width = 0
                            )
        
    def move(self, x, y, action):
        
        # Alter every cell to be the one before it (except the first)
        for i in range(len(self.parts) - 1, 0, -1):
            self.parts[i][0] = self.parts[i - 1][0]
            self.parts[i][1] = self.parts[i - 1][1]

        # If we were moving right, turning left would mean we go up, turning right would mean we go down
        clockwise = ["R", "D", "L", "U"]
        index = clockwise.index(self.current_direction)

        # Note: Move straight = [1, 0, 0], Turn left = [0, 0, 1], Turn right = [0, 1, 0]

        # Move straight
        if np.array_equal(action, [1, 0, 0]):
            self.current_direction = clockwise[index]
        
        # Turn right
        elif np.array_equal(action, [0, 1, 0]):
            self.current_direction = clockwise[(index + 1) % 4]

        # Turn left
        else:
            self.current_direction = clockwise[(index - 1) % 4]
             
        # Move the first cell in the current direction set
        match self.current_direction:

            case "L":
                self.parts[0][0] -= x
                
            case "R":
                self.parts[0][0] += x

            case "D":
                self.parts[0][1] += y

            case "U":
                self.parts[0][1] -= y
        
        # Used so that food can be generated in cells that the snake is not "occupying"
        self.occupied_cells = set((coord[0], coord[1]) for coord in self.parts)

    def check_collision(self, screen_width, screen_height, point = None):
        
        if point == None:
            point = self.parts[0]
        
        # Left, screen_width, Top, screen_height
        if point[0] < 0 or point[0] >= screen_width or point[1] < 0 or point[1] >= screen_height:
            return True
        
        # Collision with other parts
        for i in range(1, len(self.parts)):
            if point == self.parts[i]:
                return True

        return False
    
    def check_food_collision(self, food_coord):
        return True if self.parts[0][0] == food_coord[0] and self.parts[0][1] == food_coord[1] else False
    
    def extend(self, cell_size, action):
        
        last_segment = self.parts[-1][:]
        self.move(x = cell_size[0], y = cell_size[1], action = action)
        self.parts.append(last_segment)