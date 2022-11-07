from PIL import Image, ImageDraw, ImageFont
import numpy as np

from components.drone_state import DroneState
from world import ENVIRONMENT

from ml_deeco.simulation import SIMULATION_GLOBALS
from shutil import disk_usage
import sys

COLORS = {
    'drone': [0, 0, 255],
    'bird': [255, 20, 102],
    'field': [206, 215, 193],
    'charger': [204, 204, 0],
    'grid': [255, 255, 255],
    'corp': [0, 0, 0],
    'text': (0, 0, 0),
    'line': [255,0,0],
}

SIZES = {
    'drone': 4,
    'bird': 3,
    'field': 10,
    'charger': 8,
    'corp': 10,
}

LEGEND_SIZE = 400  # 260
LOWER_EXTRA = 100  # 50
TEXT_MARGIN = 15   # 20
SUB_ANIMATION_FRAMES = 1000


def check_free_space_for_animation(filename):
    if '/dev/shm/' in filename:
        mb_limit = 1000
        mb_limit = 0
        save_animation = disk_usage('/dev/shm/').free / 1024 ** 2 > mb_limit
        if not save_animation:
            print('NOT enough space for animation', file=sys.stderr)
        return not save_animation
    else:
        print('not implemented check - todo (no raise tough...)', file=sys.stderr)
        return False # problem not triggered

class Visualizer:

    def __init__(self, world):
        self.world = world
        self.cellSize = SIZES['field']
        self.width = ENVIRONMENT.mapWidth * self.cellSize + LEGEND_SIZE  # 150 for legends

        self.height = ENVIRONMENT.mapHeight * self.cellSize + LOWER_EXTRA
        try:
            self.font = ImageFont.truetype("consola.ttf", 11)
        except:
            self.font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono.ttf", 11)

        self.images = []
        self.grid = {}
        self.created = []

    def drawRectangle(self, canvas, point, component):

        startY = (point.y) * self.cellSize
        endY = startY + SIZES[component]
        startX = (point.x) * self.cellSize
        endX = startX + SIZES[component]
        for i in range(int(startY), int(endY)):
            for j in range(int(startX), int(endX)):
                canvas[i][j] = COLORS[component]
        return (startX + (endX - startX) / 2, startY + (endY - startY) / 2)

    def drawCircle(self, drawObject, rectangelMap, color):
        x1 = rectangelMap[0] * self.cellSize
        y1 = rectangelMap[1] * self.cellSize
        x2 = rectangelMap[2] * self.cellSize
        y2 = rectangelMap[3] * self.cellSize

        drawObject.ellipse((x1, y1, x2, y2), outline=color)

    def drawFields(self):
        self.background = np.zeros(
            (self.height,
             self.width, 3))  # 3 for RGB and H for unsigned short
        self.background.fill(255)
        for field in self.world.fields:
            filedPoints = field.locationPoints()
            self.grid[field] = filedPoints
            for point in filedPoints:
                self.drawRectangle(self.background, point, 'field')
        # draw a line
        legendStartPoint = self.width - LEGEND_SIZE
        for i in range(self.height):
            self.background[i][legendStartPoint] = COLORS['line']

    def getLegends(self):
        totalDamage = sum([field.damage for field in self.world.fields])
        totalCorp = sum([field.allCrops for field in self.world.fields])

        def limit_decimals(*x):
            def _limit(s):
                if type(s) is float:
                    return format(s, '.2f')
                return s

            return [_limit(a) for a in x]

        def prep(p):
            return str(limit_decimals(*p.get_the_ensml_compatible_row))

        cs = str([c.get_the_ensml_compatible_row for c in self.world.chargers])
        newtab = '\n+ '

        ds = newtab + newtab.join(
            f"{d._id} "
            f"{limit_decimals(d.get_the_ensml_compatible_row[0])[0]},"
            f"{str(d.state).split('.')[1]},"
            f"{str(d.nn_wish).split('.')[1]},"
            f"{str(d.DEBUG_nn_wish).split('.')[1]}"

            for d in self.world.drones) + '\n'

        # ds = newtab + newtab.join(prep(d) for d in self.world.drones) + '\n'
        # bs = newtab + newtab.join(prep(b) for b in self.world.birds) + '\n'

        # ds = newtab + newtab.join([str(d.get_the_ensml_compatible_row) for d in self.world.drones]) + '\n'
        # bs = newtab + newtab.join([str(b.get_the_ensml_compatible_row) for b in self.world.birds]) + '\n'

        text = f"Step: {SIMULATION_GLOBALS.currentTimeStep + 1}"

        text = f"{text}\nalive drones: {len([drone for drone in self.world.drones if drone.state != DroneState.TERMINATED])} - Damage: {totalDamage}/{totalCorp}"
        text = f"{text}\nchargers: {len(self.world.chargers)} - charger capacity: {ENVIRONMENT.chargerCapacity}"
        text = f"{text}\nbirds: {len(self.world.birds)}"
        text = f"{text}\nCharging Rate: {sum([len(charger.chargingDrones) for charger in self.world.chargers])} (drones at) {ENVIRONMENT.currentChargingRate:0.3f}"
        text = f"{text}\nMAX Charging Available: {ENVIRONMENT.totalAvailableChargingEnergy:0.3f}"

        text = f"{text}\ncharger states: {cs}"
        text = f"{text}\ndrones states: {ds}"
        # text = f"{text}\nbirds states: {bs}"

        text = f"{text}\nFields Queues:"
        for field in self.world.fields:
            text = f'{text}\n-{field.id}, #{len(field.protectingDrones)}/{len(field.places)}'
        text = f"{text}\nCharger Queues:"

        for charger in self.world.chargers:
            accepted = set(charger.acceptedDrones)
            waiting = set(charger.waitingDrones)
            potential = set(charger.potentialDrones)

            text = f"{text}\n-{charger.id}, C:{len(charger.chargingDrones)}, A:{len(accepted)}, W:{len(waiting - accepted)}, P:{len(potential - waiting - accepted)}"


            for drone in charger.chargingDrones:
                text = f"{text}\n--{drone.id}, b:{drone.battery:.2f} - C, t:{drone.timeToDoneCharging():.0f}"
            for drone in charger.acceptedDrones:
                text = f"{text}\n--{drone.id}, b:{drone.battery:.2f} - A, t:{drone.timeToDoneCharging():.0f}"
            for drone in waiting - accepted:
                text = f"{text}\n--{drone.id}, b:{drone.battery:.2f} - W, t:{drone.timeToDoneCharging():.0f}"
            for drone in potential - waiting - accepted:
                text = f"{text}\n--{drone.id}, b:{drone.battery:.2f} - P, t:{drone.timeToDoneCharging():.0f}"

        text = f"{text}\nDead Drones:"
        for drone in self.world.drones:
            if drone.state == DroneState.TERMINATED:
                text = f"{text}\n-{drone.id}"


        return text

    def drawLegends(self, draw):

        legendStartPoint = self.width - LEGEND_SIZE
        text = self.getLegends()
        draw.text((legendStartPoint + TEXT_MARGIN, TEXT_MARGIN), text, COLORS['text'],font=self.font)
        return draw

    def drawComponents(self, iteration=0, name=""):

        array = np.array(self.background, copy=True)

        for bird in self.world.birds:
            self.grid[bird] = self.drawRectangle(array, bird.location, 'bird')

        for drone in self.world.drones:
            self.grid[drone] = self.drawRectangle(array, drone.location, 'drone')

        for charger in self.world.chargers:
            self.grid[charger] = self.drawRectangle(array, charger.location, 'charger')

        for field in self.world.fields:
            for damagedCorp in field.damaged:
                self.drawRectangle(array, damagedCorp, 'corp')

        image = Image.fromarray(array.astype(np.uint8), 'RGB')
        draw = ImageDraw.Draw(image)
        for drone in self.world.drones:
            if drone.state == DroneState.TERMINATED:
                continue
            if drone.state == DroneState.CHARGING:
                color = 'red'
            elif drone.state == DroneState.PROTECTING:
                color = 'lime'
            elif drone.state == DroneState.IDLE:
                color = 'green'
            elif drone.state == DroneState.MOVING_TO_FIELD:
                color = 'blue'
            elif drone.state == DroneState.MOVING_TO_CHARGER:
                color = 'pink'

            # color = 'blue' if not drone.state == DroneState.CHARGING else 'red'
            self.drawCircle(draw, drone.protectRadius(), color)

            battery = "=0" if drone.battery == 0 else f"{drone.battery:.2f}"

            draw.text((self.grid[drone]), f"\n{drone.id}\nbattery:{battery}", COLORS['text'], font=self.font)

        for charger in self.world.chargers:
            draw.text((self.grid[charger]), f"{charger.id}", COLORS['text'], font=self.font)

        draw = self.drawLegends(draw)
        self.images.append(image)

        if len(self.images) >= SUB_ANIMATION_FRAMES:
            name = f"{name}_{len(self.created)}.gif"
            self.created.append(name)

            self.createAnimation(name, False)
            self.images = []

    def createAnimation(self, filename, final):
        if check_free_space_for_animation(filename):
            # "animation stopped"
            return
        print(f"saving {filename}")

        if not final or not self.created:
            if self.images:
                self.images[0].save(filename, save_all=True, append_images=self.images[1:])
            return

        if self.images:
            # if there are still some images (if the number of epochs is not divisible by the count of frames)
            if self.created:
                last_name = self.created[-1]
                ext = last_name[-4:]
                name = last_name[:-4] + "_last_" + ext
                print(f"saving {name}")
                self.created.append(name)
                self.images[0].save(name, save_all=True, append_images=self.images[1:])

        images = [Image.open(i) for i in self.created]
        images[0].save(filename, save_all=True, append_images=images[1:])
        self.created = []
        self.images = []
        print("final image saved")




