import pyglet
from pyglet import image
import numpy as np

WINDOW_W = 800
WINDOW_H = 600

vector = np.array


class Vertex:

    def __init__(self, pos=None, tex_coor=None, color=None, rhw=None):
        self.pos = pos if pos is not None else np.zeros(4)
        self.tex_coor = tex_coor if tex_coor is not None else np.zeros(2)
        self.color = color if color is not None else np.zeros(3)
        self.rhw = rhw if rhw else 0


class Device:

    def __init__(self, width, height):
        self.frame_buffer = image.create(width, WINDOW_H)
        self.frame_data = np.zeros((width, WINDOW_H, 4), dtype='uint8')
        self.frame_buffer.set_data(
            'RGBA', width * 4, self.frame_data.tostring())

if __name__ == '__main__':
    game_window = pyglet.window.Window(WINDOW_W, WINDOW_H)
    device = Device(WINDOW_W, WINDOW_H)

    mesh = vector([
        Vertex(pos=vector([1, -1,  1, 1]), tex_coor=vector([0, 0]),
               color=vector([1.0, 0.2, 0.2]), rhw=1),
        Vertex(pos=vector([1, -1,  1, 1]), tex_coor=vector([0, 1]),
               color=vector([0.2, 1.0, 0.2]), rhw=1),
        Vertex(pos=vector([1,  1,  1, 1]), tex_coor=vector([1, 1]),
               color=vector([0.2, 0.2, 1.0]), rhw=1),
        Vertex(pos=vector([1,  1,  1, 1]), tex_coor=vector([1, 0]),
               color=vector([1.0, 0.2, 1.0]), rhw=1),
        Vertex(pos=vector([1, -1, -1, 1]), tex_coor=vector([0, 0]),
               color=vector([1.0, 1.0, 0.2]), rhw=1),
        Vertex(pos=vector([1, -1, -1, 1]), tex_coor=vector([0, 1]),
               color=vector([0.2, 1.0, 1.0]), rhw=1),
        Vertex(pos=vector([1,  1, -1, 1]), tex_coor=vector([1, 1]),
               color=vector([1.0, 0.3, 0.3]), rhw=1),
        Vertex(pos=vector([1,  1, -1, 1]), tex_coor=vector([1, 0]),
               color=vector([0.2, 1.0, 0.3]), rhw=1)
    ])

    @game_window.event
    def on_draw():
        game_window.clear()
        device.frame_buffer.blit(0, 0)

    def update(dt):
        pass

    pyglet.clock.schedule_interval(update, 1 / 60.0)

    pyglet.app.run()
