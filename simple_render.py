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

    def copy(self):
        return Vertex(pos=self.pos.copy(),
                      tex_coor=self.tex_coor.copy(),
                      color=self.color.copy(),
                      rhw=self.rhw)


class Device:
    RENDER_STATE_WIREFRAME = 1
    RENDER_STATE_TEXTURE = 2

    def __init__(self, width, height):
        self.state = Device.RENDER_STATE_TEXTURE
        self._frame_buffer = np.zeros((height, width, 4), dtype='uint8')
        self._z_buffer = np.zeros((height, width), dtype='float')


        self.width = width
        self.height = height

        self._world_trans = np.eye(4)
        self._view_trans = np.eye(4)
        self._projection_trans = np.eye(4)
        self._trans = np.eye(4)
        self.set_perspective(
            np.pi * 0.5, float(self.width) / self.height, 1.0, 500)

    def set_texture(self, texture):
        self.texture = texture

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
        w = transformed_v[3]

        # homogenize
        transformed_v /= transformed_v[3]
        transformed_v[0] = (transformed_v[0] + 1) * self.width * 0.5
        transformed_v[1] = (1 - transformed_v[1]) * self.height * 0.5
        transformed_v[3] = w

        return transformed_v

    def clear_frame_buffer(self, color=vector([0, 0, 0, 255])):
        self._frame_buffer[..., 0] = color[0]
        self._frame_buffer[..., 1] = color[1]
        self._frame_buffer[..., 2] = color[2]
        self._frame_buffer[..., 3] = color[3]
        self._z_buffer[...] = 0

    def get_frame_buffer_str(self):
        return self._frame_buffer.tostring()

    def draw_line(self, x0, y0, x1, y1, color=vector([255,255,255,255])):
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
                self._frame_buffer[y, x, :] = color
            else:
                self._frame_buffer[x, y, :] = color
            error = error - delta_y
            if error < 0:
                y += y_step
                error += delta_x

    def _interp(self, x1, x2, t):
        return (x2-x1) * t + x1

    def _vertex_init_rhw(self, v):
        rhw = 1.0 / v.pos[3]
        v.rhw = rhw
        v.tex_coor *= rhw
        v.color *= rhw

    def _vertex_interp(self, x1, x2, t):
        res = Vertex()
        res.pos = self._interp(x1.pos, x2.pos, t)
        res.pos[3] = 1
        res.tex_coor = self._interp(x1.tex_coor, x2.tex_coor, t)
        res.color = self._interp(x1.color, x2.color, t)
        res.rhw = self._interp(x1.rhw, x2.rhw, t)
        return res


    def _trapezoid_triangle(self, v1, v2, v3):
        """
        v1, v2, v3: Vertex obj.
        ret list of trapezoid 0~2
        """
        # sort by pos[1]
        if v1.pos[1] < v2.pos[1]:
            v1, v2 = v2, v1
        if v1.pos[1] < v3.pos[1]:
            v1, v3 = v3, v1
        if v2.pos[1] < v3.pos[1]:
            v2, v3 = v3, v2

        # collinear
        if v1.pos[1] == v2.pos[1] and v2.pos[1] == v3.pos[1]:
            return []
        if v1.pos[0] == v2.pos[0] and v2.pos[0] == v3.pos[0]:
            return []

        # triangle down
        if v1.pos[1] - v2.pos[1] < 0.5:
            if v1.pos[0] > v2.pos[0]:
                v1, v2 = v2, v1
            return ((v1, v3), (v2, v3)),
        # triangle up
        if v2.pos[1] - v3.pos[1] < 0.5:
            if v2.pos[0] > v3.pos[0]:
                v2, v3 = v3, v2
            return ((v1, v2), (v1, v3)),

        t = (v2.pos[1] - v3.pos[1]) / (v1.pos[1] - v3.pos[1])

        middle = self._vertex_interp(v3, v1, t)

        if middle.pos[0] < v2.pos[0]:  # middle on the left
            return (((v1, middle), (v1, v2)),  # top tri - left edge, right edge
                    ((middle, v3), (v2, v3)))  # bottom tri - left edge, right edge
        else:
            return (((v1, v2), (v1, middle)),  # top tri - left edge, right edge
                    ((v2, v3), (middle, v3)))  # bottom tri - left edge, right edge

    def _texture_readline(self, tex, start, end, line_width, rhw):
        h, w, c = tex.shape
        h -= 1
        w -= 1

        start_tex = start.tex_coor[0] * w, start.tex_coor[1] * h
        end_tex = end.tex_coor[0] * w, end.tex_coor[1] * h

        if line_width != 0:
            x = (np.linspace(start_tex[0], end_tex[0], line_width)/rhw+0.5).astype(int)
            y = (np.linspace(start_tex[1], end_tex[1], line_width)/rhw+0.5).astype(int)

            return tex[y, x]
        else:
            return tex[start_tex[1], start_tex[0]]

    def _draw_scan_line(self, trapezoid):
        left_edge = trapezoid[0]
        right_edge = trapezoid[1]

        bottom = int(left_edge[1].pos[1] + 0.5)
        top = int(left_edge[0].pos[1] + 0.5)


        for i in xrange(bottom, top):

            t = float(i - bottom) / (top - bottom)

            start = self._vertex_interp(left_edge[0], left_edge[1], t)
            end = self._vertex_interp(right_edge[0], right_edge[1], t)

            l = int(start.pos[0] + 0.5)
            r = int(end.pos[0] + 0.5)

            cur_y = int(start.pos[1]+0.5)

            if r-l == 0:
                self._frame_buffer[cur_y, l] = self._texture_readline(self.texture, start, end, r-l, 0)
                continue
            z_buffer = self._z_buffer[cur_y, l:r]
            frame_buffer = self._frame_buffer[cur_y, l:r]
            rhw = np.linspace(start.rhw, end.rhw, r-l)

            tex_line = self._texture_readline(self.texture, start, end, r-l, rhw)

            mask = z_buffer <= rhw
            frame_buffer[mask] = tex_line[mask]
            z_buffer[mask] = rhw[mask]


    def _is_backface(self, v1, v2, v3):
        m = np.vstack((v1.pos, v2.pos, v3.pos, v1.pos))
        return np.linalg.det(m[:2, :2]) + np.linalg.det(m[1:3, :2]) + np.linalg.det(m[2:4, :2]) >= 0


    def draw_primitive(self, v1, v2, v3):

        p1 = v1.copy()
        p2 = v2.copy()
        p3 = v3.copy()

        p1.pos = device.transform(v1.pos)
        p2.pos = device.transform(v2.pos)
        p3.pos = device.transform(v3.pos)

        # backface culling
        if self._is_backface(p1, p2, p3):
            return

        # simple clip TODO: This clip is wrong. Need to be fixed
        for v in [p1, p2, p3]:
            if v.pos[1] >= device.height:
                v.pos[1] = device.height - 1
            if v.pos[1] < 0:
                v.pos[1] = 0
            if v.pos[0] >= device.width:
                v.pos[0] = device.width - 1
            if v.pos[0] < 0:
                v.pos[0] = 0
        if self.state == Device.RENDER_STATE_WIREFRAME:
            self.draw_line(p1.pos[1].astype(int), p1.pos[0].astype(int),
                           p2.pos[1].astype(int), p2.pos[0].astype(int))
            self.draw_line(p2.pos[1].astype(int), p2.pos[0].astype(int),
                           p3.pos[1].astype(int), p3.pos[0].astype(int))
            self.draw_line(p3.pos[1].astype(int), p3.pos[0].astype(int),
                           p1.pos[1].astype(int), p1.pos[0].astype(int))
        elif self.state == Device.RENDER_STATE_TEXTURE:
            self._vertex_init_rhw(p1)
            self._vertex_init_rhw(p2)
            self._vertex_init_rhw(p3)

            trapezoids = self._trapezoid_triangle(p1, p2, p3)
            for trap in trapezoids:
                self._draw_scan_line(trap)
        else:
            raise Exception("Invalid Render state %s" % self.state)


    def draw_quad(self, v1, v2, v3, v4):
        v1.tex_coor = vector([0, 0])
        v2.tex_coor = vector([0, 1])
        v3.tex_coor = vector([1, 1])
        v4.tex_coor = vector([1, 0])
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
    indices = [[0, 1, 2, 3], [7, 6, 5, 4], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]

    device.set_camera(eye=vector([0, 0, -3, 1]),
                      at=vector([0, 0, 0, 1]),
                      up=vector([0, 1, 0, 1]))

    # the rotate degree
    d = 1

    frame = image.create(device.width, device.height)
    fps_display = pyglet.clock.ClockDisplay()

    # produce texture
    texture = np.ones((256, 256, 4), dtype='uint8') * 255
    grid_size = 32
    for i in range(grid_size):
        for j in [j * 2 for j in range(grid_size / 2)]:
            texture[i * grid_size:i * grid_size + grid_size,
            (j + (i % 2)) * grid_size:(j + (i % 2)) * grid_size + grid_size, :] = vector([0, 0, 0, 255])

    device.set_texture(texture)


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
