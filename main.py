from pygame import init as pygame_init
from pygame.display import set_caption as pygame_display_set_caption
from pygame.display import update as pygame_display_update
from pygame.display import set_mode as pygame_display_set_mode
from pygame.event import get as pygame_event_get
from pygame import QUIT as pygame_QUIT
from pygame import quit as pygame_quit
from sys import exit as sys_exit
from environment import Environment
from agent import Agent

class Main:
    def __init__(self):

        # Pygame set-up
        pygame_init()

        # Set the caption
        pygame_display_set_caption("MiniMaxTicTacToe")

        # Display
        dimensions = (900, 900)
        self.screen = pygame_display_set_mode(dimensions)

        # Environment + Agent
        self.agent = Agent()
        self.environment = Environment(board_dimensions = dimensions, num_simulations = self.agent.num_simulations)
    
    def run(self):
 
        while True:
            
            # Fill screen
            self.screen.fill("WHITE")

            # ------------------------------------------------------
            # Training

            # Get old game state
            state_old = self.agent.get_state(self.environment)

            # Get the move based on the current game state
            final_move = self.agent.get_action(state_old)

            # Perform the move
            self.environment.run(action = final_move)

            state_new = self.agent.get_state(self.environment)

            # The AI collided with itself / 
            is_game_over = self.environment.reward == -10

            # Train short memory
            self.agent.train_short_memory(
                                        state = state_old, 
                                        action = final_move, 
                                        reward = self.environment.reward, 
                                        next_state = state_new, 
                                        game_over = is_game_over
                                        )
            # Add the (state, action, reward, next_state, game_over) to memory
            self.agent.remember(
                                state = state_old, 
                                action = final_move, 
                                reward = self.environment.reward, 
                                next_state = state_new, 
                                game_over = is_game_over
                                )
            
            # AI lost
            if is_game_over:
                # Train long memory
                self.agent.train_long_memory()
                self.agent.num_simulations += 1
                self.environment.num_simulations = self.agent.num_simulations
                
                # Set new high score and save the model
                if self.environment.c_score > self.environment.h_score:
                    self.environment.h_score = self.environment.c_score 
                    self.agent.model.save()

            # Set the reward back to 0
            self.environment.reward = 0
            
            # ------------------------------------------------------

            # Event handler
            self.handle_events()
            
            # Update display
            pygame_display_update() 

    def handle_events(self):

        for event in pygame_event_get():
            
            if event.type == pygame_QUIT:
                pygame_quit()
                sys_exit()

if __name__ == "__main__":
    # Instantiate main and run it
    main = Main()
    main.run()