# most recent version where things are being drawn
#this project is based off of the Code it yourself: 3D graphics pt 1 video by one lone coder, link:
import time

import pygame, math, copy, pygame.gfxdraw

from numba import jit

pygame.init()

#win = pygame.surface.Surface((1300 * 0.5, 700 * 0.5)) # lower the floats go, lower the resolution goes
win = pygame.surface.Surface((1300, 700))

window = pygame.display.set_mode((1300, 700))

# keep in mind: all variables with all caps names are acronyms

elapsed_time = 0 # must be defined before rotation matrices, instead of just some spot hidden in my spagetti, i put it here


def distance_finder(one,two) :
    [x1,y1,z1] = one  # first coordinates
    [x2,y2,z2] = two  # second coordinates

    return (((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))**(1/2)


def average(numbers):
    count = 0
    total = 0
    for n in numbers:
        count += 1
        total += n
    return total / count


def file_to_list(path_to_file):
    with open(path_to_file, "r") as level_file:
        level_file.readline()
        level_file.readline()
        text = level_file.read()
        list_of_file = []
        for i in text:
            list_of_file.append(i)
        final_list = []
        prev_index = 0
        for i in range(len(list_of_file)):
            if list_of_file[i] == " " or list_of_file[i] == "\n":
                final_list.append(text[prev_index: i])
                prev_index = i + 1
        finaliser = []
        index_1 = 0
        index_2 = 0
        for i in final_list:
            if i == "v" or i == "s" or i == "f":
                try:
                    finaliser.append([final_list[index_2], final_list[index_2+1], final_list[index_2+2], final_list[index_2+3]])
                    index_1 += 1
                except: pass
            index_2 += 1
    return finaliser


def blur_high_pixels(surface, minblur):
    # Create a new surface to hold the blurred version
    blurred_surface = pygame.Surface(surface.get_size())

    # Iterate over each pixel in the original surface
    for x in range(surface.get_width()):
        for y in range(surface.get_height()):
            # Get the RGB values for the current pixel
            r, g, b, a = surface.get_at((x, y))

            # Check if any of the RGB values are over 201
            if r > minblur or g > minblur or b > minblur:
                # Average the RGB values of the pixel and its neighbors
                total_r = total_g = total_b = 0
                count = 0

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        # Skip the center pixel
                        if dx == dy == 0:
                            continue

                        # Get the RGB values for the neighboring pixel
                        nx = x + dx
                        ny = y + dy

                        if nx < 0 or nx >= surface.get_width() or ny < 0 or ny >= surface.get_height():
                            continue

                        nr, ng, nb, na = surface.get_at((nx, ny))

                        # Add the RGB values to the running total
                        total_r += nr
                        total_g += ng
                        total_b += nb
                        count += 1

                # Calculate the average RGB values
                avg_r = total_r // count
                avg_g = total_g // count
                avg_b = total_b // count

                # Set the blurred pixel to the average RGB values
                blurred_surface.set_at((x, y), (avg_r, avg_g, avg_b, a))
            else:
                # Copy the original pixel to the blurred surface
                blurred_surface.set_at((x, y), (r, g, b, a))

    return blurred_surface



class manager:
    """
    Class for managing miscellaneous things in the program

    notes on meshes: All meshes should have a name, that's why it's a dict and not a list
    """
    meshes = {}


class math_stuff:
    """for viewing the world"""
    def __init__(self, FOV=90, Zfar=1000, Znear=0.1):
        self.FOV = FOV
        self.XY_scale = 1/math.tan(math.radians(self.FOV/2))
        self.Z_scale = (Zfar / Zfar - Znear) - (Zfar * Znear / Zfar - Znear)
        self.aspect_ratio = win.get_height() / win.get_width()
        self.Znear = Znear
        self.Zfar = Zfar


cam1 = math_stuff()

class mat4x4:
    def __init__(self, matrix=[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]):
        self.matrix = matrix

    def Make_Zeroes(self):
        return mat4x4([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    def Make_Identity(self):
        m = mat4x4([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        m.matrix[0][0] = 1.0
        m.matrix[1][1] = 1.0
        m.matrix[2][2] = 1.0
        m.matrix[3][3] = 1.0
        return m

    def Make_projection_mat(self):
        matP = mat4x4.Make_Identity(mat4x4)
        matP.matrix[0][0] = cam1.aspect_ratio
        matP.matrix[1][1] = cam1.XY_scale
        matP.matrix[2][2] = cam1.Zfar / (cam1.Zfar - cam1.Znear)
        matP.matrix[3][2] = ((cam1.Zfar * -1) * cam1.Znear / cam1.Zfar - cam1.Znear)
        matP.matrix[2][3] = 1
        matP.matrix[3][3] = 0
        return matP


    def Make_rotX_deg(self, angle): # don't ask why it's called MatRotXenos, just accept it and move on
        """
        should be:
        [1, 0,          0]
        [0, cos(angle), -sin(angle)]
        [0, sin(angle), cos(angle)]
        :param angle:
        :return:
        """
        #MatRotXenos.matrix[0][0] = math.cos(math.radians(angle))
        #MatRotXenos.matrix[0][1] = math.sin(math.radians(angle))
        #MatRotXenos.matrix[1][0] = -math.sin(math.radians(angle))
        #MatRotXenos.matrix[1][1] = math.cos(math.radians(angle))
        #MatRotXenos.matrix[2][2] = 1
        #MatRotXenos.matrix[3][3] = 1
        MatRotXenos = mat4x4.Make_Identity(mat4x4)
        MatRotXenos.matrix[0][0] = 1
        MatRotXenos.matrix[0][1] = 0
        MatRotXenos.matrix[0][2] = 0
        MatRotXenos.matrix[0][3] = 0
        MatRotXenos.matrix[1][0] = 0
        MatRotXenos.matrix[2][0] = 0
        MatRotXenos.matrix[3][0] = 0
        MatRotXenos.matrix[1][1] = math.cos(math.radians(angle))
        MatRotXenos.matrix[1][2] = -math.sin(math.radians(angle))
        MatRotXenos.matrix[2][1] = math.sin(math.radians(angle))
        MatRotXenos.matrix[2][2] = math.cos(math.radians(angle))
        return MatRotXenos

    def Make_rotZ_deg(self, angle): # this is also returning nan
        MatRotZ = mat4x4.Make_Identity(mat4x4)
        MatRotZ.matrix[0][0] = 1
        MatRotZ.matrix[1][1] = math.cos(math.radians(angle))
        MatRotZ.matrix[1][2] = math.sin(math.radians(angle))
        MatRotZ.matrix[2][1] = -math.sin(math.radians(angle))
        MatRotZ.matrix[2][2] = math.cos(math.radians(angle))
        MatRotZ.matrix[3][3] = 1
        return MatRotZ

    def Make_rotY_deg(self, angle):
        matRotY = mat4x4()
        matRotY.matrix[0][0] = math.cos(math.radians(angle))
        matRotY.matrix[0][2] = math.sin(math.radians(angle))
        matRotY.matrix[2][0] = -math.sin(math.radians((angle)))
        matRotY.matrix[2][1] = 1
        matRotY.matrix[2][2] = math.cos(math.radians(angle))
        matRotY.matrix[3][3] = 1
        return matRotY

    def Make_Translation(self, x, y, z):
        matT = mat4x4.Make_Identity(mat4x4)
        matT.matrix[0][0] = 1.0
        matT.matrix[1][1] = 1.0
        matT.matrix[2][2] = 1.0
        matT.matrix[3][3] = 1.0
        matT.matrix[3][0] = x
        matT.matrix[3][1] = y
        matT.matrix[3][2] = z
        return matT

    def matXmat(self, other): #only ever returns a mat4x4 filled with zeroes
        # based off of the first answer in: https://www.programiz.com/python-programming/examples/multiply-matrix
        result = mat4x4.Make_Zeroes(mat4x4)
        for i in range(len(self.matrix)):
            # iterate through columns of Y
            for j in range(len(other.matrix[0])):
                # iterate through rows of Y
                for k in range(len(other.matrix)):
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j]
        return result

    def mat_PointAt(self, posV, targetV, upV):
        newForward = vec3d.sub(targetV, posV)
        newForward = vec3d.normalise(newForward)

        a = vec3d.__mul__(newForward, vec3d.dot_product(upV, newForward))
        newUp = vec3d.sub(upV, a)
        newUp = vec3d.normalise(newUp)

        newRight = vec3d.cross_product(newUp, newForward)

        returnval = mat4x4.Make_Zeroes(mat4x4)
        returnval.matrix[0][0] = newRight.x
        returnval.matrix[0][1] = newRight.y
        returnval.matrix[0][2] = newRight.z
        returnval.matrix[0][3] = 0.0
        returnval.matrix[1][0] = newUp.x
        returnval.matrix[1][1] = newUp.y
        returnval.matrix[1][2] = newUp.z
        returnval.matrix[1][3] = 0.0
        returnval.matrix[2][0] = newForward.x
        returnval.matrix[2][1] = newForward.y
        returnval.matrix[2][2] = newForward.z
        returnval.matrix[2][3] = 0.0
        returnval.matrix[3][0] = posV.x
        returnval.matrix[3][1] = posV.y
        returnval.matrix[3][2] = posV.z
        returnval.matrix[3][3] = 1.0
        return returnval

    def Matrix_QuickInverse(m): # Only for Rotation/Translation Matrices
        inversed = mat4x4.Make_Zeroes(mat4x4)
        returnval = mat4x4.Make_Zeroes(mat4x4)
        returnval.matrix[0][0] = m.matrix[0][0]
        returnval.matrix[0][1] = m.matrix[1][0]
        returnval.matrix[0][2] = m.matrix[2][0]
        returnval.matrix[0][3] = 0.0
        returnval.matrix[1][0] = m.matrix[0][1]
        returnval.matrix[1][1] = m.matrix[1][1]
        returnval.matrix[1][2] = m.matrix[2][1]
        returnval.matrix[1][3] = 0.0
        returnval.matrix[2][0] = m.matrix[0][2]
        returnval.matrix[2][1] = m.matrix[1][2]
        returnval.matrix[2][2] = m.matrix[2][2]
        returnval.matrix[2][3] = 0.0
        returnval.matrix[3][0] = -(m.matrix[3][0] * returnval.matrix[0][0] + m.matrix[3][1] * returnval.matrix[1][0] + m.matrix[3][2] * returnval.matrix[2][0])
        returnval.matrix[3][1] = -(m.matrix[3][0] * returnval.matrix[0][1] + m.matrix[3][1] * returnval.matrix[1][1] + m.matrix[3][2] * returnval.matrix[2][1])
        returnval.matrix[3][2] = -(m.matrix[3][0] * returnval.matrix[0][2] + m.matrix[3][1] * returnval.matrix[1][2] + m.matrix[3][2] * returnval.matrix[2][2])
        returnval.matrix[3][3] = 1.0
        return returnval


def mult_Vect_Mat(vec, mat):
    op = vec3d(x=0, y=0, z=0)
    op.x = vec.x * mat.matrix[0][0] + vec.y * mat.matrix[1][0] + vec.z * mat.matrix[2][0] + vec.w * mat.matrix[3][0]
    op.y = vec.x * mat.matrix[0][1] + vec.y * mat.matrix[1][1] + vec.z * mat.matrix[2][1] + vec.w * mat.matrix[3][1]
    op.z = vec.x * mat.matrix[0][2] + vec.y * mat.matrix[1][2] + vec.z * mat.matrix[2][2] + vec.w * mat.matrix[3][2]
    op.w = vec.x * mat.matrix[0][3] + vec.y * mat.matrix[1][3] + vec.z * mat.matrix[2][3] + vec.w * mat.matrix[3][3]
    try:
        op.x /= op.w
        op.y /= op.w
        op.z /= op.w
    except:
        #op = 1 # if this is triggered, clip function likely malfunctioned
       op.x = -1 # if this is triggered, clip function likely malfunctioned
       op.y = -1 # if this is triggered, clip function likely malfunctioned
       op.z = -1 # if this is triggered, clip function likely malfunctioned
       op.a = "E" # if this is triggered, clip function likely malfunctioned
    return op

# projection matrix setup
matP = mat4x4.Make_projection_mat(mat4x4)

#matRotX = mat4x4.Make_rotX_deg(mat4x4, 0)
#matRotY = mat4x4.Make_rotY_deg(mat4x4, 0)
#matRotZ = mat4x4.Make_rotZ_deg(mat4x4, 0)

# Thanks to Rabbid76 on StackOverflow (https://stackoverflow.com/questions/6339057/draw-a-transparent-rectangles-and-polygons-in-pygame)
# for this function!
def draw_polygon_alpha(surface, color, points):
    lx, ly = zip(*points)
    min_x, min_y, max_x, max_y = min(lx), min(ly), max(lx), max(ly)
    target_rect = pygame.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.polygon(shape_surf, color, [(x - min_x, y - min_y) for x, y in points])
    surface.blit(shape_surf, target_rect)


def draw_triangle_2D(surf, colour, p1, p2, p3):
    pygame.draw.line(surf, colour, p1, p2, 3)
    pygame.draw.line(surf, colour, p2, p3, 3)
    pygame.draw.line(surf, colour, p3, p1, 3)


# classes related to basic 3D stuff

vLightSources = []
class vec3d:
    """
    A 3D vector for storing values in 3D space
    """
    def __init__(self, x, y, z, w=1, islight=False):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)
        if islight: vLightSources.append(self.normalise())


    def __str__(self):
        return f"(x: {self.x}, y: {self.y}, z: {self.z})"

    def __add__(self, other):
        if type(other) == vec3d:
            return vec3d(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            return vec3d(self.x + other, self.y + other, self.z + other)

    def sub(self, other): # for whatever reason, if this was a magic method it would return an int no matter what
        if type(other) == type(self):
            return vec3d(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            return vec3d(self.x - other, self.y - other, self.z - other)

    def __mul__(self, other):
        return vec3d(self.x * other, self.y * other, self.z * other)

    def vecDIVvec(self, other):
        return vec3d(self.x / other.x, self.y / other.y, self.z / other.z, self.w / other.w)

    def vecDIVfloat(self, other):
        return vec3d(self.x / other, self.y / other, self.z / other, self.w / other)

    def dot_product(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def length(self):
        return math.sqrt(self.dot_product(self))

    def normalise(self):
        l = self.length()
        return vec3d(self.x / l, self.y / l, self.z / l)

    def cross_product(self, other):
        returnval = vec3d(0, 0, 0)
        returnval.x = self.y * other.z - self.z * other.y
        returnval.y = self.z * other.x - self.x * other.z
        returnval.z = self.x * other.y - self.y * other.x
        return returnval

    #def Vector_IntersectPlane(self, plane_p, plane_n, lineStart, lineEnd):
    #    # comments documenting things being 0 only accounts for the first time this function  is ran, as that's when excecution stops
    #    #print(plane_p, plane_n, lineStart, lineEnd)
    #    plane_n = vec3d.normalise(plane_n)
    #    plane_d = vec3d.dot_product(plane_n, plane_p) * -1
    #    ad = vec3d.dot_product(lineStart, plane_n) # always 0, check some equations on the c++ one to see what's up
    #    bd = vec3d.dot_product(lineEnd, plane_n) # always 0, check some equations on the c++ one to see what's up
    #    #print(bd)
    #    #print(ad)
    #    try:
    #        t = plane_d - ad / bd - ad # bd - ad is 0
    #    except:
    #        t = 2 # bd - ad is 0
    #    lineStartToEnd = vec3d.sub(lineEnd, lineStart)
    #    lineToIntersect = vec3d.__mul__(lineStartToEnd, t)
    #    return vec3d.__add__(lineStart, lineToIntersect)

    def Vector_IntersectPlane(self, plane_p, plane_n, lineStart, lineEnd):
        if vec3d.magnitude(vec3d.sub(lineEnd, lineStart)) == 0:
            #print(lineStart)
            return 0

        plane_n = vec3d.normalise(plane_n)  # normalize the plane normal vector
        plane_d = vec3d.dot_product(plane_n, plane_p) * -1  # calculate the plane distance from the origin

        t = (plane_d - vec3d.dot_product(plane_n, lineStart)) / vec3d.dot_product(plane_n, vec3d.sub(lineEnd, lineStart))
        #ad = vec3d.dot_product(lineStart, plane_n)  # calculate the dot product between the line start and plane normal
        #bd = vec3d.dot_product(lineEnd, plane_n)  # calculate the dot product between the line end and plane normal

        #t = (plane_d - ad) / (bd - ad)  # calculate the intersection point parameter along the line

        lineStartToEnd = vec3d.sub(lineEnd, lineStart)  # calculate the vector from line start to end
        lineToIntersect = vec3d.__mul__(lineStartToEnd, t)  # calculate the vector from line start to intersection point

        return vec3d.__add__(lineStart, lineToIntersect)  # return the intersection point

    def Triangle_ClipAgainstPlane(self, plane_p, plane_n, in_tri): # out_tri1 and out_tri2 are being returned in a list.
        plane_n = vec3d.normalise(plane_n)
        def dist(p):
            n = vec3d.normalise(p)
            return (plane_n.x * p.x + plane_n.y * p.y + plane_n.z * p.z - vec3d.dot_product(plane_n, plane_p))

        inside_points = [vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0)]
        nInsidePointCount = 0
        outside_points = [vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0)]
        nOutsidePointCount = 0

        d0 = dist(in_tri.vec1)
        d1 = dist(in_tri.vec2)
        d2 = dist(in_tri.vec3)

        if d0 >= 0:
            nInsidePointCount += 1
            inside_points[nInsidePointCount] = in_tri.vec1
        else:
            nOutsidePointCount += 1
            outside_points[nOutsidePointCount] = in_tri.vec1
        if d1 >= 0:
            nInsidePointCount += 1
            inside_points[nInsidePointCount] = in_tri.vec2
        else:
            nOutsidePointCount += 1
            outside_points[nOutsidePointCount] = in_tri.vec2
        if d2 >= 0:
            nInsidePointCount += 1
            inside_points[nInsidePointCount] = in_tri.vec3
        else:
            nOutsidePointCount += 1
            outside_points[nOutsidePointCount] = in_tri.vec3

        # whole triangle in/out
        if nInsidePointCount == 0: # working fine
            # whole triangle out of plane
            out_tri1 = copy.deepcopy(in_tri)
            out_tri2 = copy.deepcopy(in_tri)
            return [0, out_tri1, out_tri2]
        if nInsidePointCount == 3: # working fine
            # whole thing in plane
            out_tri1 = copy.deepcopy(in_tri)
            out_tri2 = copy.deepcopy(in_tri)
            return [1, copy.deepcopy(in_tri), out_tri2]

        # part in, part out
        if nInsidePointCount == 1 and nOutsidePointCount == 2: # one vec3d is off
            #print("1 new")
            out_tri1 = copy.deepcopy(in_tri)
            out_tri2 = copy.deepcopy(in_tri)

            out_tri1.vec1 = inside_points[0]
            out_tri1.vec2 = vec3d.Vector_IntersectPlane(vec3d, plane_p, plane_n, inside_points[0], outside_points[0]) # something goes wrong here
            out_tri1.vec3 = vec3d.Vector_IntersectPlane(vec3d, plane_p, plane_n, inside_points[0], outside_points[1]) # something goes wrong here
            if out_tri1.vec2 == 0 or out_tri1.vec3 == 0:
                return 0
            return [1, out_tri1, out_tri2]

        if nInsidePointCount == 2 and nOutsidePointCount == 1: # two vec3ds are not right
            #print("2 new")
            out_tri1 = copy.deepcopy(in_tri)
            out_tri2 = copy.deepcopy(in_tri)

            out_tri1.vec1 = inside_points[0]
            out_tri1.vec2 = inside_points[1]
            out_tri1.vec3 = vec3d.Vector_IntersectPlane(vec3d, plane_p, plane_n, inside_points[0], outside_points[0]) # something goes wrong here

            out_tri2.vec1 = inside_points[1]
            out_tri2.vec2 = out_tri1.vec3
            out_tri2.vec3 = vec3d.Vector_IntersectPlane(vec3d, plane_p, plane_n, inside_points[1], outside_points[0]) # something goes wrong here

            if in_tri.vec3 == 0 or out_tri1.vec2 == 0 or out_tri1.vec3 == 0:
                return 0
            return [1, out_tri1, out_tri2]

    # written by chatgpt
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)



vCamera = vec3d(0, 0, 0)
manager.vLookDir = vec3d.normalise(vec3d(0, 1, 0))
manager.vOgDir = vec3d.normalise(vec3d(0, 1, 0))


class triangle:
    def __init__(self, vec3d1, vec3d2, vec3d3, col=False):
        self.vec1 = vec3d1
        self.vec2 = vec3d2
        self.vec3 = vec3d3
        self.col = col

    def Make_Blanc(self):
        return triangle(vec3d(0, 0, 0), vec3d(0, 0, 0), vec3d(0, 0, 0))

    def get_distance_ratio(triangle, point): # written (mostly) by chatGPT!
        """
        Calculates the distance between the average point of a triangle and a vec3d,
        then scales it between 0 and 1 based on the maximum distance in the scene.
        """
        # Find the average point of the triangle
        center = vec3d(
            (triangle.vec1.x + triangle.vec2.x + triangle.vec3.x) / 3,
            (triangle.vec1.y + triangle.vec2.y + triangle.vec3.y) / 3,
            (triangle.vec1.z + triangle.vec2.z + triangle.vec3.z) / 3
        )

        # Calculate the distance between the center point and the given point
        distance = vec3d.sub(center, point).magnitude()

        # Scale the distance between 0 and 1 based on the maximum distance in the scene
        max_distance = 5  # replace with the maximum distance in your scene
        distance_ratio = min(distance / max_distance, 1)

        return distance_ratio

def sort_key(t):
    return t[0].vec1.z + t[0].vec2.z + t[0].vec3.z // 3

class mesh:
    def __init__(self, triangles=[], name="mesh"):
        self.members = triangles
        manager.meshes.update({name: self})


    def load_OBJ_file(self, sFilename):
        """
        takes a .obj file and uses the vertex data to create a mesh
        :param sFilename: path to file
        :return: none
        """
        verts = [] # to be filled with vertex data from the obj
        tris = []

        list_of_info = file_to_list(sFilename)

        for i in list_of_info:
            if i[0] == "v":
                i.pop(0)
                for o in i:
                    o = float(o)
                verts.append(vec3d(x=i[0], y=i[1], z=i[2]))
            if i[0] == "f":
                i.pop(0)
                for o in i:
                    o = float(o)
                tris.append(triangle(vec3d1=verts[int(i[0]) - 1], vec3d2=verts[int(i[1]) - 1], vec3d3=verts[int(i[2]) - 1]))
        mesh(tris, sFilename[0:-4])

    def change_X(self, amount):
        for t in self.members:
            t.change_X(amount)

    def change_Y(self, amount):
        for t in self.members:
            t.change_Y(amount)

    def change_Z(self, amount):
        for t in self.members:
            t.change_Z(amount)

    def draw(self):
        matRotX = mat4x4.Make_rotX_deg(mat4x4, 0)
        matRotZ = mat4x4.Make_rotZ_deg(mat4x4, 0)
        #matRotX = mat4x4.Make_rotX_deg(mat4x4, elapsed_time)
        #matRotZ = mat4x4.Make_rotZ_deg(mat4x4, elapsed_time)
        triangles = []
        for t in self.members:
            # rotation
            triViewed = triangle.Make_Blanc(triangle)
            matTrans = mat4x4.Make_Translation(mat4x4, 0, 0, 16)
            matWorld = mat4x4.matXmat(matRotZ, matRotX)
            matWorld = mat4x4.matXmat(matWorld, matTrans) # translation

            #camera work, I need to add a visualisation of the direction the camera is facing
            manager.vLookDir = manager.vOgDir
            vUp = vec3d(0, 1, 0)
            vTarget = vec3d(0, 0, 1)
            matCameraRot = mat4x4.Make_rotY_deg(mat4x4, camRoll)
            #matCameraRot2 = mat4x4.Make_rotY_deg(mat4x4, camYaw)
            matCameraRot2 = matCameraRot
            matCameraRot3 = mat4x4.Make_rotZ_deg(mat4x4, camUD)

            matTotalRot1 = mat4x4.matXmat(matCameraRot, matCameraRot2)
            matTotalRot2 = mat4x4.matXmat(matTotalRot1, matCameraRot3)

            manager.vLookDir = mult_Vect_Mat(vTarget, matTotalRot2)
            vTarget = vCamera + manager.vLookDir

            matCamera = mat4x4.mat_PointAt(mat4x4, vCamera, vTarget, vUp)

            matView = mat4x4.Matrix_QuickInverse(matCamera)

            # transformation of triangles
            tri_transformed = triangle(vec3d(0, 0, 0, 1), vec3d(0, 0, 0, 1), vec3d(0, 0, 0, 1))
            tri_transformed.vec1 = mult_Vect_Mat(t.vec1, matWorld)
            tri_transformed.vec2 = mult_Vect_Mat(t.vec2, matWorld)
            tri_transformed.vec3 = mult_Vect_Mat(t.vec3, matWorld)

            ## calculating normals
            line1 = vec3d.sub(tri_transformed.vec2, tri_transformed.vec1)
            line2 = vec3d.sub(tri_transformed.vec3, tri_transformed.vec1)

            normal = vec3d.cross_product(line1, line2)

            normal = vec3d.normalise(normal)

            # projection 3D --> 2D
            vCameraRay = vec3d.sub(tri_transformed.vec1, vCamera)

            if vec3d.dot_product(normal, vCameraRay) < 0:
                # lighting
                total_dots = []
                for l in vLightSources:
                    dot_light_surf = triangle.get_distance_ratio(tri_transformed, l)
                    total_dots.append(dot_light_surf)

                for l in range(len(total_dots)):
                    total_dots[l] *= vec3d.dot_product(normal, vLightSources[l])

                try:
                    #dot_light_surf = average(total_dots) doesnt look nice but helps visualise which triangles are getting the most light
                    dot_light_surf = max(total_dots)
                except: dot_light_surf = 0
                # worldspace to viewedspace
                triViewed.vec1 = mult_Vect_Mat(tri_transformed.vec1, matView)
                triViewed.vec2 = mult_Vect_Mat(tri_transformed.vec2, matView)
                triViewed.vec3 = mult_Vect_Mat(tri_transformed.vec3, matView)

                # clipping

                clipped = [triangle.Make_Blanc(triangle), triangle.Make_Blanc(triangle)]

                 #                                          self,        plane_p (clip)          plane_n,       in_tri,     out_tri1, out_tri2
                ClipInfo = vec3d.Triangle_ClipAgainstPlane(vec3d, vec3d(0.0, 0.0, 0.1), vec3d(0.0, 0.0, 1.0), triViewed) # requires pointers, moved them to the return list
                if type(ClipInfo) == type(0): continue
                nClippedTriangles = ClipInfo[0]
                clipped[0] = ClipInfo[1]
                clipped[1] = ClipInfo[2]

                for i in range(nClippedTriangles):

                    # remove before release. if not removed, rest assured this is just for debugging the clipping of triangles
                    if mult_Vect_Mat(clipped[i].vec1, matP) == 1 or\
                        mult_Vect_Mat(clipped[i].vec2, matP) == 1 or\
                        mult_Vect_Mat(clipped[i].vec3, matP) == 1:
                        continue
                    try:
                        print(clipped[i].a)
                        print(clipped[i])
                    except:
                        tri_pojected = triangle(
                            mult_Vect_Mat(clipped[i].vec1, matP),
                            mult_Vect_Mat(clipped[i].vec2, matP),
                            mult_Vect_Mat(clipped[i].vec3, matP))

                        tri_pojected.vec1.x += 1
                        tri_pojected.vec1.y += 1
                        tri_pojected.vec2.x += 1
                        tri_pojected.vec2.y += 1
                        tri_pojected.vec3.x += 1
                        tri_pojected.vec3.y += 1

                        tri_pojected.vec1.x *= 0.5 * win.get_width()
                        tri_pojected.vec2.x *= 0.5 * win.get_width()
                        tri_pojected.vec3.x *= 0.5 * win.get_width()

                        tri_pojected.vec1.y *= 0.5 * win.get_height()
                        tri_pojected.vec2.y *= 0.5 * win.get_height()
                        tri_pojected.vec3.y *= 0.5 * win.get_height()

                        # ensure nothing is too bright
                        if dot_light_surf > 1:
                            dot_light_surf = 0.9
                        elif dot_light_surf < 0:
                            dot_light_surf = 0

                        triangles.append([tri_pojected, dot_light_surf * triangle.get_distance_ratio(tri_pojected, vCamera)])

        # out of the loop of calculated for every triangle
        triangles.sort(key=sort_key)
        triangles.reverse()

        for t in triangles:
            # actual_drawing
            pygame.draw.polygon(win, (235 * t[1] + 20, 235 * t[1] + 20, 235 * t[1] + 20), [(t[0].vec1.x, t[0].vec1.y), (t[0].vec2.x, t[0].vec2.y), (t[0].vec3.x, t[0].vec3.y)])
            #draw_triangle_2D(win, (235 * t[1] + 20, 235 * t[1] + 20, 235 * t[1] + 20), (t[0].vec1.x, t[0].vec1.y), (t[0].vec2.x, t[0].vec2.y), (t[0].vec3.x, t[0].vec3.y))

    def draw_camera_visualisation(self):
        matRotX = mat4x4.Make_rotX_deg(mat4x4, 0)
        matRotZ = mat4x4.Make_rotZ_deg(mat4x4, 0)
        # rotation
        triViewed = triangle.Make_Blanc(triangle)
        matTrans = mat4x4.Make_Translation(mat4x4, 0, 0, 16)
        matWorld = mat4x4.matXmat(matRotZ, matRotX)
        matWorld = mat4x4.matXmat(matWorld, matTrans)  # translation

        # camera work, I need to add a visualisation of the direction the camera is facing
        manager.vLookDir = manager.vOgDir
        vUp = vec3d(0, 1, 0)
        vTarget = vec3d(0, 0, 1)
        matCameraRot = mat4x4.Make_rotY_deg(mat4x4, camRoll)
        # matCameraRot2 = mat4x4.Make_rotY_deg(mat4x4, camYaw)
        matCameraRot2 = matCameraRot
        matCameraRot3 = mat4x4.Make_rotZ_deg(mat4x4, camUD)

        matTotalRot1 = mat4x4.matXmat(matCameraRot, matCameraRot2)
        matTotalRot2 = mat4x4.matXmat(matTotalRot1, matCameraRot3)

        manager.vLookDir = mult_Vect_Mat(vTarget, matTotalRot2)
        vTarget = vCamera + manager.vLookDir

        matCamera = mat4x4.mat_PointAt(mat4x4, vCamera, vTarget, vUp)

        matView = mat4x4.Matrix_QuickInverse(matCamera)

        # transformation of triangles
        tri_transformed = triangle(vec3d(0, 0, 0, 1), vec3d(0, 0, 0, 1), vec3d(0, 0, 0, 1))
        tri_transformed.vec1 = mult_Vect_Mat(t.vec1, matWorld)
        tri_transformed.vec2 = mult_Vect_Mat(t.vec2, matWorld)
        tri_transformed.vec3 = mult_Vect_Mat(t.vec3, matWorld)

        ## calculating normals
        line1 = vec3d.sub(tri_transformed.vec2, tri_transformed.vec1)
        line2 = vec3d.sub(tri_transformed.vec3, tri_transformed.vec1)

        normal = vec3d.cross_product(line1, line2)

        normal = vec3d.normalise(normal)

        # projection 3D --> 2D
        vCameraRay = vec3d.sub(tri_transformed.vec1, vCamera)

        if vec3d.dot_product(normal, vCameraRay) < 0:
            # lighting
            total_dots = []
            for l in vLightSources:
                dot_light_surf = triangle.get_distance_ratio(tri_transformed, l)
                total_dots.append(dot_light_surf)

            for l in range(len(total_dots)):
                total_dots[l] *= vec3d.dot_product(normal, vLightSources[l])

            try:
                # dot_light_surf = average(total_dots) doesnt look nice but helps visualise which triangles are getting the most light
                dot_light_surf = max(total_dots)
            except:
                dot_light_surf = 0
            # worldspace to viewedspace
            triViewed.vec1 = mult_Vect_Mat(tri_transformed.vec1, matView)
            triViewed.vec2 = mult_Vect_Mat(tri_transformed.vec2, matView)
            triViewed.vec3 = mult_Vect_Mat(tri_transformed.vec3, matView)

            # clipping

            tri_pojected = triangle(
                mult_Vect_Mat(clipped[i].vec1, matP),
                mult_Vect_Mat(clipped[i].vec2, matP),
                mult_Vect_Mat(clipped[i].vec3, matP))

            tri_pojected.vec1.x += 1
            tri_pojected.vec1.y += 1
            tri_pojected.vec2.x += 1
            tri_pojected.vec2.y += 1
            tri_pojected.vec3.x += 1
            tri_pojected.vec3.y += 1

            tri_pojected.vec1.x *= 0.5 * win.get_width()
            tri_pojected.vec2.x *= 0.5 * win.get_width()
            tri_pojected.vec3.x *= 0.5 * win.get_width()

            tri_pojected.vec1.y *= 0.5 * win.get_height()
            tri_pojected.vec2.y *= 0.5 * win.get_height()
            tri_pojected.vec3.y *= 0.5 * win.get_height()

            # ensure nothing is too bright
            if dot_light_surf > 1:
                dot_light_surf = 0.9
            elif dot_light_surf < 0:
                dot_light_surf = 0

            triangles.append(
                [tri_pojected, dot_light_surf * triangle.get_distance_ratio(tri_pojected, vCamera)])

        pygame.draw.polygon(win, (235 * t[1] + 20, 235 * t[1] + 20, 235 * t[1] + 20),
                                [(t[0].vec1.x, t[0].vec1.y), (t[0].vec2.x, t[0].vec2.y), (t[0].vec3.x, t[0].vec3.y)])


# with the basic 3D stuff out of the way, it's time for the stuff more useful should one make a actual game with this


def on_run():
    global vCamera
    vCamera = vec3d(0, 0, 0)

    #mesh.load_OBJ_file(mesh, "models\\sphere.obj")
    #mesh.load_OBJ_file(mesh, "models\\better sphere.obj")
    #mesh.load_OBJ_file(mesh, "models\\scene.obj")
    #mesh.load_OBJ_file(mesh, "models\\teapootis.obj")
    #mesh.load_OBJ_file(mesh, "models\\XYZ.obj")
    mesh.load_OBJ_file(mesh, "models\\tree field.obj")
    #mesh.load_OBJ_file(mesh, "models\\scene house.obj")
    #mesh.load_OBJ_file(mesh, "models\\light tests.obj")
    #mesh.load_OBJ_file(mesh, "models\\sort test(icles).obj")
    #mesh.load_OBJ_file(mesh, "models\\best sphere.obj")
    #mesh.load_OBJ_file(mesh, "models\\DONT OPEN.obj")
    #mesh.load_OBJ_file(mesh, "models\\.obj")

    #sun = vec3d.normalise(vec3d(0, 0, -1, islight=True))


def redrawgamewindow():
    win.fill((0, 0, 0))
    for m in manager.meshes.values():
        m.draw()
    window_todraw = pygame.transform.scale(win, (window.get_width(), window.get_height()))
    #window_todraw = blur_high_pixels(window_todraw, 50)
    window.blit(window_todraw, (0, 0))
    pygame.display.update()


on_run() # only ever run once


run = True
clock = pygame.time.Clock()
camYaw = 0
camUD = 0
camRoll = 0
lightsOn = True
while run:
    clock.tick(300)

    redrawgamewindow()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()

    elapsed_time += 1
    if keys[pygame.K_LCTRL]: elapsed_time += 1
    if keys[pygame.K_LSHIFT]: elapsed_time -= 1

    if keys[pygame.K_ESCAPE]:
        run = False
    # cam controls
    if keys[pygame.K_LEFT]:
        camYaw += 2
    if keys[pygame.K_RIGHT]:
        camYaw -= 2
    if keys[pygame.K_UP]:
        camUD -= 2
    if keys[pygame.K_DOWN]:
        camUD += 2

    vForward = vec3d.__mul__(manager.vLookDir, 0.5)

    # movement
    if keys[pygame.K_r]:
        vCamera.y -= 1
    if keys[pygame.K_f]:
        vCamera.y += 1
    if keys[pygame.K_d]:
        vCamera.x += math.cos(math.radians(camYaw))
        vCamera.z += math.sin(math.radians(camYaw))
    if keys[pygame.K_a]:
        vCamera.x -= math.cos(math.radians(camYaw))
        vCamera.z -= math.sin(math.radians(camYaw))
    if keys[pygame.K_w]:
        vCamera += vForward
    if keys[pygame.K_s]:
        vCamera = vec3d.sub(vCamera, vForward)# would be -= but there were issues with vec3d.__sub__() so this had to do
    if keys[pygame.K_q]:
        camRoll += 2
    if keys[pygame.K_e]:
        camRoll -= 2
    if keys[pygame.K_x]:
        lightsOn = not lightsOn
    if keys[pygame.K_z]:
        vec3d(vForward.x * -1, vForward.y * -1, vForward.z * -1, islight=True)
    if lightsOn:
        vLightSources.clear()
        vec3d(vForward.x * -1, vForward.y * -1, vForward.z * -1, islight=True)
    if keys[pygame.K_c]:
        vLightSources.clear()
    if keys[pygame.K_p]: # reset camera
        camYaw = 0
        camRoll = 0
        camUD = 0

    # debugging
    if keys[pygame.K_SPACE]:
        print(clock.get_fps())

pygame.quit()
