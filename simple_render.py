import pyglet
from pyglet import image
import numpy as np

WINDOW_W = 800
WINDOW_H = 600


def vector(x):
    return np.array(x, dtype='float32')


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

        self._world_trans = np.eye(4)
        self._view_trans = np.eye(4)
        self._projection_trans = np.eye(4)
        self._trans = np.eye(4)
        self.set_perspective(
            np.pi * 0.5, float(self.width) / self.height, 1.0, 500)

    @staticmethod
    def make_transform(scale=None, rotate=None, translate=None):
        """
        scale: (scalex, scaley, scalez)
        rotate: (x, y, z, theta). <x, y, z> indicate the axis rotated by,
                theta is the degree of rotation
        translate: (translatex, translatey, translatez)
        """
        res = np.eye(4)

        if scale:
            tmp_matrix = np.eye(4)
            tmp_matrix[0, 0] = scale[0]
            tmp_matrix[1, 1] = scale[1]
            tmp_matrix[2, 2] = scale[2]
            res = np.dot(res, tmp_matrix)

        if rotate:
            tmp_matrix = np.eye(4)
            x, y, z = Device._normalize(vector(rotate[:3]))
            theta = rotate[3]

            cos = np.cos(theta)
            one_sub_cos = 1 - cos
            sin = np.sin(theta)
            tmp_matrix[0, 0] = cos + one_sub_cos * x ** 2
            tmp_matrix[0, 1] = one_sub_cos * x * y - sin * z
            tmp_matrix[0, 2] = one_sub_cos * x * z + sin * y
            tmp_matrix[1, 0] = one_sub_cos * y * x + sin * z
            tmp_matrix[1, 1] = cos + one_sub_cos * y ** 2
            tmp_matrix[1, 2] = one_sub_cos * y * z - sin * x
            tmp_matrix[2, 0] = one_sub_cos * z * x - sin * y
            tmp_matrix[2, 1] = one_sub_cos * z * y + sin * x
            tmp_matrix[2, 2] = cos + one_sub_cos * z ** 2
            res = np.dot(res, tmp_matrix)

        if translate:
            tmp_matrix = np.eye(4)
            tmp_matrix[0, 3] = translate[0]
            tmp_matrix[1, 3] = translate[1]
            tmp_matrix[2, 3] = translate[2]
            res = np.dot(res, tmp_matrix)

        return res

    @staticmethod
    def _normalize(vec):
        n = np.linalg.norm(vec)
        return vec / n if n != 0 else vec

    def set_camera(self, eye, at, up):

        z = self._normalize((at - eye)[:3])
        x = self._normalize(np.cross(up[:3], z))
        y = self._normalize(np.cross(z, x))

        tmp_matrix = np.eye(4)
        tmp_matrix[0, :3] = x
        tmp_matrix[1, :3] = y
        tmp_matrix[2, :3] = z
        tmp_matrix[3, 0] = -np.dot(x, eye[:3])
        tmp_matrix[3, 1] = -np.dot(y, eye[:3])
        tmp_matrix[3, 2] = -np.dot(z, eye[:3])

        self._view_trans = tmp_matrix
        self.update_transform()

    def set_perspective(self, fov, aspect, zn, zf):
        tmp_matrix = np.zeros((4, 4))
        cot = 1. / np.tan(fov / 2.)
        tmp_matrix[0, 0] = cot / aspect
        tmp_matrix[1, 1] = cot
        tmp_matrix[2, 2] = zf / (zf - zn)
        tmp_matrix[2, 3] = 1
        tmp_matrix[3, 2] = zf * zn / (zn - zf)

        self._projection_trans = tmp_matrix
        self.update_transform()

    def set_world_trans(self, transform):
        self._world_trans = transform
        self.update_transform()

    def update_transform(self):
        tmp_trans = np.dot(self._world_trans, self._view_trans)
        self._trans = np.dot(tmp_trans, self._projection_trans)

    def transform(self, v):
        # transform
        transformed_v = np.dot(v, self._trans)

        # homogenize
        transformed_v /= transformed_v[3]
        transformed_v[0] = (transformed_v[0] + 1) * self.width * 0.5
        transformed_v[1] = (1 - transformed_v[1]) * self.height * 0.5

        return transformed_v

    def clear_frame_buffer(self, color=vector([0, 0, 0, 255])):
        self._frame_buffer[..., 0] = color[0]
        self._frame_buffer[..., 1] = color[1]
        self._frame_buffer[..., 2] = color[2]
        self._frame_buffer[..., 3] = color[3]

    def get_frame_buffer_str(self):
        return self._frame_buffer.tostring()

    def draw_line(self, x0, y0, x1, y1):
        """
        Bresenham line drawing
        x0, y0, x1, y1 must be integer
        """
        steep = np.abs(y1 - y0) > np.abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        delta_x = x1 - x0
        delta_y = np.abs(y1 - y0)
        error = delta_x / 2

        y = y0
        if y0 < y1:
            y_step = 1
        else:
            y_step = -1

        for x in xrange(x0, x1):
            if steep:
                self._frame_buffer[y, x, :] = 255
            else:
                self._frame_buffer[x, y, :] = 255
            error = error - delta_y
            if error < 0:
                y += y_step
                error += delta_x

    def draw_primitive(self, v1, v2, v3):
        v1_pos = device.transform(v1.pos)
        v2_pos = device.transform(v2.pos)
        v3_pos = device.transform(v3.pos)

        # simple clip
        for v_pos in [v1_pos, v2_pos, v3_pos]:
            if v_pos[1] >= device.height:
                v_pos[1] = device.height - 1
            if v_pos[1] < 0:
                v_pos[1] = 0
            if v_pos[0] >= device.width:
                v_pos[0] = device.width - 1
            if v_pos[0] < 0:
                v_pos[0] = 0
        self.draw_line(v1_pos[1].astype(int), v1_pos[0].astype(int),
                       v2_pos[1].astype(int), v2_pos[0].astype(int))
        self.draw_line(v2_pos[1].astype(int), v2_pos[0].astype(int),
                       v3_pos[1].astype(int), v3_pos[0].astype(int))
        self.draw_line(v3_pos[1].astype(int), v3_pos[0].astype(int),
                       v1_pos[1].astype(int), v1_pos[0].astype(int))

    def draw_quad(self, v1, v2, v3, v4):
        self.draw_primitive(v1, v2, v3)
        self.draw_primitive(v3, v4, v1)

    def draw_mesh(self, vertices, indices):
        for i in indices:
            if len(i) == 3:
                self.draw_primitive(vertices[i[0]], vertices[i[1]], vertices[i[2]])
            else:
                self.draw_quad(vertices[i[0]], vertices[i[1]], vertices[i[2]], vertices[i[3]])


if __name__ == '__main__':
    game_window = pyglet.window.Window(WINDOW_W, WINDOW_H)
    device = Device(WINDOW_W, WINDOW_H)

    # define the mesh of box
    mesh = [
        Vertex(pos=vector([1, -1, 1, 1]), tex_coor=vector([0, 0]),
               color=vector([1.0, 0.2, 0.2]), rhw=1),
        Vertex(pos=vector([-1, -1, 1, 1]), tex_coor=vector([0, 1]),
               color=vector([0.2, 1.0, 0.2]), rhw=1),
        Vertex(pos=vector([-1, 1, 1, 1]), tex_coor=vector([1, 1]),
               color=vector([0.2, 0.2, 1.0]), rhw=1),
        Vertex(pos=vector([1, 1, 1, 1]), tex_coor=vector([1, 0]),
               color=vector([1.0, 0.2, 1.0]), rhw=1),
        Vertex(pos=vector([1, -1, -1, 1]), tex_coor=vector([0, 0]),
               color=vector([1.0, 1.0, 0.2]), rhw=1),
        Vertex(pos=vector([-1, -1, -1, 1]), tex_coor=vector([0, 1]),
               color=vector([0.2, 1.0, 1.0]), rhw=1),
        Vertex(pos=vector([-1, 1, -1, 1]), tex_coor=vector([1, 1]),
               color=vector([1.0, 0.3, 0.3]), rhw=1),
        Vertex(pos=vector([1, 1, -1, 1]), tex_coor=vector([1, 0]),
               color=vector([0.2, 1.0, 0.3]), rhw=1)
    ]
    indices = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]

    frame = image.create(device.width, device.height)
    device.set_camera(eye=vector([3, 0, 0, 1]),
                      at=vector([0, 0, 0, 1]),
                      up=vector([0, 0, 1, 1]))

    d = 1

    fps_display = pyglet.clock.ClockDisplay()


    @game_window.event
    def on_draw():
        game_window.clear()
        device.clear_frame_buffer()

        global d
        trans = device.make_transform(rotate=(1, 1, 1, d / np.pi * 180))
        device.set_world_trans(trans)
        d += 0.0005
        d %= 180

        # draw box
        device.draw_mesh(mesh, indices)

        frame.set_data(
            'RGBA', device.width * 4, device.get_frame_buffer_str())
        frame.blit(0, 0)
        fps_display.draw()


    def update(dt):
        pass


    pyglet.clock.schedule_interval(update, 1 / 60.0)

    pyglet.app.run()
