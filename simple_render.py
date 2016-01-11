import pyglet
from pyglet import image
import numpy as np

WINDOW_W = 800
WINDOW_H = 600


class Device:

    def __init__(self, width, height):
        self.frame_buffer = image.create(width, WINDOW_H)
        self.frame_data = np.zeros((width, WINDOW_H, 4), dtype='uint8')
        self.frame_buffer.set_data(
            'RGBA', width * 4, self.frame_data.tostring())

if __name__ == '__main__':
    game_window = pyglet.window.Window(WINDOW_W, WINDOW_H)
    device = Device(WINDOW_W, WINDOW_H)

    @game_window.event
    def on_draw():
        game_window.clear()
        device.frame_buffer.blit(0, 0)

    def update(dt):
        pass

    pyglet.clock.schedule_interval(update, 1 / 60.0)

    pyglet.app.run()
