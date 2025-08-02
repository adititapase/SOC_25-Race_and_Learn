import pygame, random, sys
from pygame.math import Vector2

pygame.init()
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))
clock = pygame.time.Clock()

game_font = pygame.font.Font('Font/PoetsenOne-Regular.ttf', 25)
apple = pygame.image.load('Graphics/apple.png').convert_alpha()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class SNAKE:
    def __init__(self):
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(1,0)
        self.new_block = False

        self.head_up = pygame.image.load('Graphics/head_up.png').convert_alpha()
        self.head_down = pygame.image.load('Graphics/head_down.png').convert_alpha()
        self.head_right = pygame.image.load('Graphics/head_right.png').convert_alpha()
        self.head_left = pygame.image.load('Graphics/head_left.png').convert_alpha()

        self.tail_up = pygame.image.load('Graphics/tail_up.png').convert_alpha()
        self.tail_down = pygame.image.load('Graphics/tail_down.png').convert_alpha()
        self.tail_right = pygame.image.load('Graphics/tail_right.png').convert_alpha()
        self.tail_left = pygame.image.load('Graphics/tail_left.png').convert_alpha()

        self.body_vertical = pygame.image.load('Graphics/body_vertical.png').convert_alpha()
        self.body_horizontal = pygame.image.load('Graphics/body_horizontal.png').convert_alpha()

        self.body_tr = pygame.image.load('Graphics/body_tr.png').convert_alpha()
        self.body_tl = pygame.image.load('Graphics/body_tl.png').convert_alpha()
        self.body_br = pygame.image.load('Graphics/body_br.png').convert_alpha()
        self.body_bl = pygame.image.load('Graphics/body_bl.png').convert_alpha()

        self.crunch_sound = pygame.mixer.Sound('Sound/crunch.wav')

    def draw_snake(self):
        self.update_head_graphics()
        self.update_tail_graphics()

        for index,block in enumerate(self.body):
            x_pos = int(block.x * cell_size)
            y_pos = int(block.y * cell_size)
            block_rect = pygame.Rect(x_pos,y_pos,cell_size,cell_size)

            if index == 0:
                screen.blit(self.head,block_rect)
            elif index == len(self.body) - 1:
                screen.blit(self.tail,block_rect)
            else:
                previous_block = self.body[index + 1] - block
                next_block = self.body[index - 1] - block
                if previous_block.x == next_block.x:
                    screen.blit(self.body_vertical,block_rect)
                elif previous_block.y == next_block.y:
                    screen.blit(self.body_horizontal,block_rect)
                else:
                    if previous_block.x == -1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == -1:
                        screen.blit(self.body_tl,block_rect)
                    elif previous_block.x == -1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == -1:
                        screen.blit(self.body_bl,block_rect)
                    elif previous_block.x == 1 and next_block.y == -1 or previous_block.y == -1 and next_block.x == 1:
                        screen.blit(self.body_tr,block_rect)
                    elif previous_block.x == 1 and next_block.y == 1 or previous_block.y == 1 and next_block.x == 1:
                        screen.blit(self.body_br,block_rect)

    def update_head_graphics(self):
        head_relation = self.body[1] - self.body[0]
        if head_relation == Vector2(1,0): self.head = self.head_left
        elif head_relation == Vector2(-1,0): self.head = self.head_right
        elif head_relation == Vector2(0,1): self.head = self.head_up
        elif head_relation == Vector2(0,-1): self.head = self.head_down

    def update_tail_graphics(self):
        tail_relation = self.body[-2] - self.body[-1]
        if tail_relation == Vector2(1,0): self.tail = self.tail_left
        elif tail_relation == Vector2(-1,0): self.tail = self.tail_right
        elif tail_relation == Vector2(0,1): self.tail = self.tail_up
        elif tail_relation == Vector2(0,-1): self.tail = self.tail_down

    def move_snake(self):
        if self.new_block:
            body_copy = self.body[:]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]
            self.new_block = False
        else:
            body_copy = self.body[:-1]
            body_copy.insert(0, body_copy[0] + self.direction)
            self.body = body_copy[:]

    def add_block(self):
        self.new_block = True

    def play_crunch_sound(self):
        self.crunch_sound.play()

    def reset(self):
        self.body = [Vector2(5,10),Vector2(4,10),Vector2(3,10)]
        self.direction = Vector2(1,0)
        self.new_block = False

class FRUIT:
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x * cell_size),int(self.pos.y * cell_size),cell_size,cell_size)
        screen.blit(apple,fruit_rect)

    def randomize(self):
        self.x = random.randint(0,cell_number - 1)
        self.y = random.randint(0,cell_number - 1)
        self.pos = Vector2(self.x,self.y)

class SnakeEnv:
    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()
        self.frame_count = 0

    def reset(self):
        self.snake.reset()
        self.fruit.randomize()
        self.frame_count = 0
        return self.get_state()

    def play_step(self, action):
        self.frame_count += 1
        self.move(action)
        self.snake.move_snake()

        reward = 0
        done = False

        if self.check_collision():
            reward = -10
            done = True
            return reward, done

        if self.snake.body[0] == self.fruit.pos:
            self.snake.add_block()
            self.snake.play_crunch_sound()
            reward = 10
            self.fruit.randomize()

        if self.frame_count > 100 * len(self.snake.body):
            done = True

        return reward, done

    def move(self, action):
        clockwise = [Vector2(1,0), Vector2(0,1), Vector2(-1,0), Vector2(0,-1)]
        idx = clockwise.index(self.snake.direction)

        if action == 0:  # straight
            new_dir = clockwise[idx]
        elif action == 1:  # right
            new_dir = clockwise[(idx + 1) % 4]
        else:  # left
            new_dir = clockwise[(idx - 1) % 4]

        self.snake.direction = new_dir

    def check_collision(self):
        head = self.snake.body[0]
        if not 0 <= head.x < cell_number or not 0 <= head.y < cell_number:
            return True
        if head in self.snake.body[1:]:
            return True
        return False

    def get_state(self):
        head = self.snake.body[0]
        point_l = head + self.left_direction()
        point_r = head + self.right_direction()
        point_s = head + self.snake.direction

        danger_left = self.collision(point_l)
        danger_right = self.collision(point_r)
        danger_straight = self.collision(point_s)

        dir_l = self.snake.direction == self.left_direction()
        dir_r = self.snake.direction == self.right_direction()
        dir_u = self.snake.direction == self.up_direction()
        dir_d = self.snake.direction == self.down_direction()

        food_left = self.fruit.pos.x < head.x
        food_right = self.fruit.pos.x > head.x
        food_up = self.fruit.pos.y < head.y
        food_down = self.fruit.pos.y > head.y

        return (
            int(danger_left), int(danger_right), int(danger_straight),
            int(dir_l), int(dir_r), int(dir_u), int(dir_d),
            int(food_left), int(food_right), int(food_up), int(food_down)
        )

    def collision(self, point):
        if not 0 <= point.x < cell_number or not 0 <= point.y < cell_number:
            return True
        if point in self.snake.body[1:]:
            return True
        return False

    def left_direction(self):
        clockwise = [Vector2(1,0), Vector2(0,1), Vector2(-1,0), Vector2(0,-1)]
        idx = clockwise.index(self.snake.direction)
        return clockwise[(idx - 1) % 4]

    def right_direction(self):
        clockwise = [Vector2(1,0), Vector2(0,1), Vector2(-1,0), Vector2(0,-1)]
        idx = clockwise.index(self.snake.direction)
        return clockwise[(idx + 1) % 4]

    def up_direction(self):
        return Vector2(0, -1)

    def down_direction(self):
        return Vector2(0, 1)

    def render(self):
        screen.fill((175, 215, 70))
        self.draw_grass()
        self.fruit.draw_fruit()
        self.snake.draw_snake()
        self.draw_score()
        pygame.display.update()
        clock.tick(60)

    def draw_grass(self):
        grass_color = (167,209,61)
        for row in range(cell_number):
            for col in range(cell_number):
                if (row + col) % 2 == 0:
                    grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                    pygame.draw.rect(screen, grass_color, grass_rect)

    def draw_score(self):
        score_text = str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (56,74,12))
        score_x = int(cell_size * cell_number - 60)
        score_y = int(cell_size * cell_number - 40)
        score_rect = score_surface.get_rect(center = (score_x, score_y))
        bg_rect = pygame.Rect(score_rect.left - 6, score_rect.top, score_rect.width + 12, score_rect.height)

        pygame.draw.rect(screen, (167,209,61), bg_rect)
        screen.blit(score_surface, score_rect)
        pygame.draw.rect(screen, (56,74,12), bg_rect, 2)
