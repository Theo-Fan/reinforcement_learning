import pygame
import sys
import random
import numpy as np

# 定义一些常量
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
GRID_SIZE = 20
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 200, 50)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GRAY = (200, 200, 200)
BROWN = (150, 75, 0)

class HarvestGame:
    def __init__(self, num_players=3):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Harvest Game")

        self.clock = pygame.time.Clock()

        self.players = []
        for i in range(num_players):
            self.players.append({
                'pos': [random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)],
                'color': [random.randint(0, 255) for _ in range(3)],
                'score': 0
            })

        # 初始化苹果
        self.apples = []

        self.init_apples()

    def init_apples(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if random.random() < 0.2:
                    self.apples.append([x, y])

    def regenerate_apples(self):
        new_apples = []
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if [x, y] not in self.apples:
                    if random.random() < 0.01:  # 再生概率
                        new_apples.append([x, y])
        self.apples.extend(new_apples)

    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (SCREEN_WIDTH, y))

    def draw_elements(self):
        # 绘制苹果
        for apple in self.apples:
            rect = pygame.Rect(apple[0]*CELL_SIZE, apple[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)

        # 绘制玩家
        for player in self.players:
            rect = pygame.Rect(player['pos'][0]*CELL_SIZE, player['pos'][1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, player['color'], rect)

    def move_player(self, player):
        # 随机移动
        action = random.choice(['up', 'down', 'left', 'right'])
        x, y = player['pos']

        if action == 'up' and y > 0:
            player['pos'][1] -= 1
        elif action == 'down' and y < GRID_SIZE - 1:
            player['pos'][1] += 1
        elif action == 'left' and x > 0:
            player['pos'][0] -= 1
        elif action == 'right' and x < GRID_SIZE - 1:
            player['pos'][0] += 1

    def check_collisions(self, player):
        # 检查是否在苹果位置
        if player['pos'] in self.apples:
            self.apples.remove(player['pos'])
            player['score'] += 1

    def run(self):
        running = True
        while running:
            self.clock.tick(10)
            self.screen.fill(WHITE)
            self.draw_grid()
            self.draw_elements()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # 更新苹果
            self.regenerate_apples()

            # 更新玩家状态
            for player in self.players:
                self.move_player(player)
                self.check_collisions(player)

            pygame.display.flip()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = HarvestGame()
    game.run()
