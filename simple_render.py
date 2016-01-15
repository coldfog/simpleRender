import pyglet
from pyglet import image
import numpy as np
import copy

# TODO mock for line drawing, delete later
from skimage.draw import line

WINDOW_W = 800
WINDOW_H = 600

vector = lambda x: np.array(x, dtype='float32')


class Vertex:

    def __init__(self, pos=None, tex_coor=None, color=None, rhw=None):
        self.pos = pos if pos is not None else np.zeros(4)
        self.tex_coor = tex_coor if tex_coor is not None else np.zeros(2)
        self.color = color if color is not None else np.zeros(3)
        self.rhw = rhw if rhw else 0


def _get_martix(scale=None, rotate=None, translate=None):
    '''
    scale: (scalex, scaley, scalez)
    rotate: (x, y, z, theta). <x, y, z> indicate the axis rotated by,
            theta is the degree of rotation
    translate: (translatex, translatey, translatez)
    '''
    res = np.eye(4)

    if scale:
        tmp_matrix = np.eye(4)
        tmp_matrix[0, 0] = scale[0]
        tmp_matrix[1, 1] = scale[1]
        tmp_matrix[2, 2] = scale[2]
        res = np.dot(res, tmp_matrix)

    if rotate:
        tmp_matrix = np.eye(4)
        x, y, z = _normalize(vector(rotate[:3]))
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


def _normalize(v):
    n = np.linalg.norm(v)
    if n != 0:
        return v / n
    else:
        return v


class Device:

    def __init__(self, width, height):
        self._frame_buffer = np.zeros((height, width, 4), dtype='uint8')

        self.width = width
        self.height = height

        self._world_trans = np.eye(4)
        self._view_trans = np.eye(4)
        self.set_perspective(
            180 * 0.5,  float(self.width) / self.height, 1.0, 500)

    def set_camera(self, eye, at, up):

        z = _normalize((at - eye)[:3])
        x = _normalize(np.cross(up[:3], z))
        y = _normalize(np.cross(z, x))

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

    def draw_primitive(self, v1, v2, v3):
        v1 = copy.deepcopy(v1)
        v2 = copy.deepcopy(v2)
        v3 = copy.deepcopy(v3)
        v1.pos = device.transform(v1.pos)
        v2.pos = device.transform(v2.pos)
        v3.pos = device.transform(v3.pos)
        # TODO mock for line drawing, implement by self later
        # simple clip
        for v in [v1, v2, v3]:
            if v.pos[1] >= device.width:
                v.pos[1] = device.width - 1
            if v.pos[1] < 0:
                v.pos[1] = 0
            if v.pos[0] >= device.height:
                v.pos[0] = device.height - 1
            if v.pos[0] < 0:
                v.pos[0] = 0
        rr, cc = line(v1.pos[0].astype(int), v1.pos[1].astype(int),
                      v2.pos[0].astype(int), v2.pos[1].astype(int))
        self._frame_buffer[rr, cc, :] = 255
        rr, cc = line(v2.pos[0].astype(int), v2.pos[1].astype(int),
                      v3.pos[0].astype(int), v3.pos[1].astype(int))
        self._frame_buffer[rr, cc, :] = 255
        rr, cc = line(v3.pos[0].astype(int), v3.pos[1].astype(int),
                      v1.pos[0].astype(int), v1.pos[1].astype(int))
        self._frame_buffer[rr, cc, :] = 255

    def draw_quad(self, v1, v2, v3, v4):
        self.draw_primitive(v1, v2, v3)
        self.draw_primitive(v3, v4, v1)


if __name__ == '__main__':
    game_window = pyglet.window.Window(WINDOW_W, WINDOW_H)
    device = Device(WINDOW_W, WINDOW_H)

    # define the mesh of box
    mesh = [
        Vertex(pos=vector([1, -1,  1, 1]), tex_coor=vector([0, 0]),
               color=vector([1.0, 0.2, 0.2]), rhw=1),
        Vertex(pos=vector([-1, -1,  1, 1]), tex_coor=vector([0, 1]),
               color=vector([0.2, 1.0, 0.2]), rhw=1),
        Vertex(pos=vector([-1,  1,  1, 1]), tex_coor=vector([1, 1]),
               color=vector([0.2, 0.2, 1.0]), rhw=1),
        Vertex(pos=vector([1,  1,  1, 1]), tex_coor=vector([1, 0]),
               color=vector([1.0, 0.2, 1.0]), rhw=1),
        Vertex(pos=vector([1, -1, -1, 1]), tex_coor=vector([0, 0]),
               color=vector([1.0, 1.0, 0.2]), rhw=1),
        Vertex(pos=vector([-1, -1, -1, 1]), tex_coor=vector([0, 1]),
               color=vector([0.2, 1.0, 1.0]), rhw=1),
        Vertex(pos=vector([-1,  1, -1, 1]), tex_coor=vector([1, 1]),
               color=vector([1.0, 0.3, 0.3]), rhw=1),
        Vertex(pos=vector([1,  1, -1, 1]), tex_coor=vector([1, 0]),
               color=vector([0.2, 1.0, 0.3]), rhw=1)
    ]

    frame = image.create(device.width, device.height)
    device.set_camera(eye=vector([3, 0, 0, 1]),
                      at=vector([0, 0, 0, 1]),
                      up=vector([0, 0, 1, 1]))

    d = 1

    @game_window.event
    def on_draw():
        game_window.clear()
        device.clear_frame_buffer()

        global d
        device.set_world_trans(_get_martix(rotate=(1, 1, 0, d)))
        d += 0.05
        d %= 180
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
