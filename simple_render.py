import pyglet
from pyglet import image
import numpy as np

# TODO mock for line drawing, delete later
from skimage.draw import line

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
        self._frame_buffer = np.zeros((height, width, 4), dtype='uint8')
        self.width = width
        self.height = height

    def clear_frame_buffer(self, color=vector([0, 0, 0, 255])):
        self._frame_buffer[..., 0] = color[0]
        self._frame_buffer[..., 1] = color[1]
        self._frame_buffer[..., 2] = color[2]
        self._frame_buffer[..., 3] = color[3]

    def get_frame_buffer_str(self):
        return self._frame_buffer.tostring()

    def draw_primitive(self, v1, v2, v3):
        # TODO mock for line drawing, implement by self later
        rr, cc = line(v1.pos[0], v1.pos[1], v2.pos[0], v2.pos[1])
        self._frame_buffer[rr, cc, :] = 255

    def draw_quad(self, v1, v2, v3, v4):
        self.draw_primitive(v1, v2, v3)
        self.draw_primitive(v3, v4, v1)


if __name__ == '__main__':
    game_window = pyglet.window.Window(WINDOW_W, WINDOW_H)
    device = Device(WINDOW_W, WINDOW_H)

    # define the mesh of box
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

    frame = image.create(device.width, device.height)

    @game_window.event
    def on_draw():
        game_window.clear()

        # draw box
        device.draw_quad(mesh[0], mesh[1], mesh[2], mesh[3])
        device.draw_quad(mesh[4], mesh[5], mesh[6], mesh[7])
        device.draw_quad(mesh[0], mesh[4], mesh[5], mesh[1])
        device.draw_quad(mesh[1], mesh[5], mesh[6], mesh[2])
        device.draw_quad(mesh[2], mesh[6], mesh[7], mesh[3])
        device.draw_quad(mesh[3], mesh[7], mesh[4], mesh[0])

        frame.set_data(
            'RGBA', device.width * 4, device.get_frame_buffer_str())
        frame.blit(0, 0)

    def update(dt):
        pass

    pyglet.clock.schedule_interval(update, 1 / 60.0)

    pyglet.app.run()
