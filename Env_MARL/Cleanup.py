import pygame
import sys
import random

# 游戏窗口和网格参数
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 600
GRID_SIZE = 30
CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

# 颜色定义
BLACK = (0, 0, 0)
RIVER_COLOR = (52, 127, 148)
APPLE_COLOR = (140, 247, 74)
GARBAGE_COLOR = (92, 74, 55)
PLAYER_COLORS = [(255, 20, 147), (255, 255, 0), (148, 0, 211)]
GRAY = (100, 100, 100)

# 游戏参数
MAX_STEPS = 1000
GARBAGE_THRESHOLD = 10
GARBAGE_LIMIT = 50
APPLE_GROWTH_RATE = 15


class CleanupGame:
    def __init__(self, num_players=3):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Cleanup Game")
        self.clock = pygame.time.Clock()

        # 初始化游戏状态
        self.players = [{'pos': self.random_position(),
                         'color': PLAYER_COLORS[i], 'score': 0}
                        for i in range(num_players)]
        self.river_area = {(x, y) for x in range(GRID_SIZE // 3) for y in range(GRID_SIZE)}
        self.resource_area = {(x, y) for x in range(GRID_SIZE // 3, GRID_SIZE) for y in range(GRID_SIZE)}
        self.apples = set()
        self.garbage = set()
        self.step_count = 0
        self.apple_growth_timer = 0
        self.game_over = False

        # 初始生成苹果
        self.init_apples()

    def random_position(self):
        return random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)

    def init_apples(self):
        for _ in range(5):
            self.apples.add(random.choice(list(self.resource_area)))

    def draw_elements(self):
        # 绘制河道和资源区域
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = RIVER_COLOR if (x, y) in self.river_area else BLACK
                pygame.draw.rect(self.screen, color, rect)

        # 绘制苹果和垃圾
        for apple in self.apples:
            pygame.draw.rect(self.screen, APPLE_COLOR,
                             pygame.Rect(apple[0] * CELL_SIZE, apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for garbage in self.garbage:
            pygame.draw.rect(self.screen, GARBAGE_COLOR,
                             pygame.Rect(garbage[0] * CELL_SIZE, garbage[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # 绘制玩家
        for player in self.players:
            pygame.draw.rect(self.screen, player['color'],
                             pygame.Rect(player['pos'][0] * CELL_SIZE, player['pos'][1] * CELL_SIZE, CELL_SIZE,
                                         CELL_SIZE))

    def move_player(self, player):
        actions = ['up', 'down', 'left', 'right', 'clean', 'stay']
        action = random.choice(actions)
        x, y = player['pos']

        if action == 'up' and y > 0:
            player['pos'] = (x, y - 1)
        elif action == 'down' and y < GRID_SIZE - 1:
            player['pos'] = (x, y + 1)
        elif action == 'left' and x > 0:
            player['pos'] = (x - 1, y)
        elif action == 'right' and x < GRID_SIZE - 1:
            player['pos'] = (x + 1, y)
        elif action == 'clean' and player['pos'] in self.garbage:
            self.garbage.remove(player['pos'])

    def generate_garbage(self):
        if len(self.garbage) < GARBAGE_LIMIT:
            self.garbage.add(random.choice(list(self.river_area)))

    def generate_apples(self):
        if len(self.garbage) <= GARBAGE_THRESHOLD:
            self.apple_growth_timer += 1
            if self.apple_growth_timer >= APPLE_GROWTH_RATE:
                available_positions = self.resource_area - self.apples
                if available_positions:
                    self.apples.add(random.choice(list(available_positions)))
                self.apple_growth_timer = 0

    def run(self):
        while not self.game_over:
            self.clock.tick(5)
            self.screen.fill(BLACK)
            self.draw_elements()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.step_count += 1
            if self.step_count % 10 == 0:
                self.generate_garbage()
            self.generate_apples()

            for player in self.players:
                self.move_player(player)
                if player['pos'] in self.apples:
                    self.apples.remove(player['pos'])
                    player['score'] += 1

            if self.step_count >= MAX_STEPS or len(self.garbage) >= GARBAGE_LIMIT:
                self.game_over = True
                print("游戏结束！")

            pygame.display.flip()


if __name__ == "__main__":
    game = CleanupGame()
    game.run()
