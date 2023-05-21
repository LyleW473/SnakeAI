from pygame.draw import line as pygame_draw_line
from pygame.draw import rect as pygame_draw_rect
from pygame.display import get_surface as pygame_display_get_surface

from pygame.font import SysFont as pygame_font_SysFont
from pygame.time import get_ticks as pygame_time_get_ticks
from pygame.key import get_pressed as pygame_key_get_pressed
from pygame import K_SPACE as pygame_K_SPACE

from snake import SnakeAI
from random import choice as random_choice

from os.path import exists as os_path_exists


class Environment:

    def __init__(self, board_dimensions, num_simulations):


        self.surface = pygame_display_get_surface()
        self.dimensions = board_dimensions
        self.num_cells = 20
        self.cell_dimensions = (self.dimensions[0] / self.num_cells, self.dimensions[1] / self.num_cells)
        self.num_simulations = num_simulations
        

        # High score
        if os_path_exists("high_score.txt"):
            with open("high_score.txt", "r") as hs_file:
                self.h_score = int(hs_file.read())
        else:
            self.h_score = 0

        self.c_score = 0

        # Text
        self.text_font = pygame_font_SysFont("Bahnschrift", 50)
        self.start_text_size = self.text_font.size("Press Spacebar to start!")

        self.reset_game(model = None)

        # AI
        self.reward = 0
        self.simulation_started = False

    def draw_text(self, text, text_colour, font, x, y):
        
        # Render the text as an image without anti-aliasing
        text_image = font.render(text, False, text_colour)
        # Blit the image onto the surface
        self.surface.blit(text_image, (x, y))

    def draw_grid(self):
        
        # Vertical
        for i in range(1, self.num_cells):
            pygame_draw_line(
                            surface = self.surface,
                            color = "BLACK",
                            start_pos = (i * self.cell_dimensions[0], 0),
                            end_pos = (i * self.cell_dimensions[0], self.dimensions[1]),
                            width = 1
                            )
        # Horizontal
        for i in range(1, self.num_cells):
            pygame_draw_line(
                            surface = self.surface,
                            color = "BLACK",
                            start_pos = (0, i * self.cell_dimensions[1]),
                            end_pos = (self.dimensions[0], i * self.cell_dimensions[1]),
                            width = 1
                            )
    
    def reset_game(self, model):

        # Determines whether the snake will extend on the next self.move() call
        self.extend_snake = None

        # Spawn snake
        self.snake = SnakeAI(
                            x = ((self.num_cells / 2) - 1) * self.cell_dimensions[0], 
                            y = ((self.num_cells / 2) - 1) * self.cell_dimensions[1]
                            )

        # Generate food
        self.food = self.generate_food()
        
        # Score 
        if self.c_score > self.h_score and model != None:
            # Save new high score
            self.h_score = self.c_score
            with open("high_score.txt", "w") as hs_file:
                hs_file.write(str(self.h_score))
            
            # Save the model
            model.save()

        # Reset the score (This is so that we can save the model and then reset the score)
        self.c_score = 0

        # Text
        self.cs_text_size = self.text_font.size(f"Score: {self.c_score}")
        self.hs_text_size = self.text_font.size(f"High score: {self.h_score}")
        self.simulations_text_size = self.text_font.size(f"No.Simulations: {self.num_simulations}")

        # Number of
        self.frame_iteration = 0

    
    def generate_food(self):

        # Used so that food can be generated in cells that the snake is not "occupying"
        self.food_locations = [(i, j) for j in range(self.num_cells) for i in range(self.num_cells) if ((i * self.cell_dimensions[0], j * self.cell_dimensions[1]) not in self.snake.occupied_cells)]
        choice = random_choice(self.food_locations)
        return (choice[0] * self.cell_dimensions[0], choice[1] * self.cell_dimensions[1])

    def run(self, action):
        
        # Draw grid
        self.draw_grid()
        
        if self.simulation_started:

            # Draw the food
            pygame_draw_rect(
                            surface = self.surface, 
                            color = "GREEN",
                            rect = (self.food[0], 
                                    self.food[1], 
                                    self.cell_dimensions[0], 
                                    self.cell_dimensions[1]
                                    ),
                            width = 0
                            )
            
            # Check if the snake collided with food
            if self.snake.check_food_collision(food_coord = self.food) == True:
                self.food = self.generate_food()
                self.extend_snake = True # Extend the snake the next time the snake moves

                # Set the reward as positive
                self.reward = 10

                # Score
                self.c_score += 1
                self.cs_text_size = self.text_font.size(f"Score: {self.c_score}") # Find new text size (for accurate alignment of text)

            
            # Move the snake
            current_time = pygame_time_get_ticks()
            if current_time - self.snake.move_timer > 10: #150:
                
                # Extend the snake
                if self.extend_snake == True:
                    self.snake.extend(cell_size = self.cell_dimensions, action = action)
                    self.extend_snake = False

                # Just move the snake
                else:
                    self.snake.move(x = self.cell_dimensions[0], y = self.cell_dimensions[1], action = action)
                    self.frame_iteration += 1

                # Reset next iteration timer
                self.snake.move_timer = current_time

            # Check if the snake has collided with the borders of the screen or with itself
            if self.snake.check_collision(screen_width = self.dimensions[0], screen_height = self.dimensions[1]) == True or self.frame_iteration > 100 * len(self.snake.parts):
                # Set the reward as negative
                self.reward = -10

            # Draw snake
            self.snake.draw(
                        surface = self.surface, 
                        width = self.cell_dimensions[0], 
                        height = self.cell_dimensions[1]
                        )
            
            # Draw the current score text
            self.draw_text(
                            text = f"Score: {self.c_score}",
                            text_colour = "BLACK",
                            font = self.text_font,
                            x = (self.dimensions[0]) - (self.cs_text_size[0]) - 40,
                            y = (self.dimensions[1]) - (self.cs_text_size[1]) - 40,
                            )
            # Draw the high score text
            self.draw_text(
                            text = f"High score: {self.h_score}",
                            text_colour = "BLACK",
                            font = self.text_font,
                            x = 20,
                            y = (self.dimensions[1]) - (self.hs_text_size[1]) - 40,
                            )
            
            # Draw number of simulations
            self.draw_text(
                            text = f"No.Simulations: {self.num_simulations}",
                            text_colour = "BLACK",
                            font = self.text_font,
                            x = 20,
                            y = 20,
                            )
            
            
        else:
            # Draw the start text
            self.draw_text(
                            text = "Press Spacebar to start!",
                            text_colour = "BLACK",
                            font = self.text_font,
                            x = (self.dimensions[0] // 2) - (self.start_text_size[0] // 2),
                            y = (self.dimensions[1] // 2) - (self.start_text_size[1] // 2)
                            )

            # Draw the high-score text
            self.draw_text(
                            text = f"High score: {self.h_score}",
                            text_colour = "BLACK",
                            font = self.text_font,
                            x = (self.dimensions[0] // 2) - (self.hs_text_size[0] // 2),
                            y = (self.dimensions[1] // 2) - (self.hs_text_size[1] // 2) + 80
                            )
            # Space to start the simulation
            if pygame_key_get_pressed()[pygame_K_SPACE]:
                self.simulation_started = True