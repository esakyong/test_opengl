import glfw
import glfw.GLFW as GLFW_CONSTANTS
from OpenGL.GL import *
import numpy as np
import ctypes
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from PIL import Image
import math


###################################################################################
# Specify a relative path
###################################################################################

import os
import sys

def get_application_path():
    if getattr(sys, 'frozen', False): # pyinstaller 등으로 묶였는지 확인
        application_path = os.path.dirname(sys.executable)

    else:
        application_path = os.path.dirname((__file__))
    return application_path

os.chdir(get_application_path())

###################################################################################
# The main codes
###################################################################################

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
RETURN_ACTION_CONTINUE = 0
RETURN_ACTION_END = 1

def initialize_glfw():

    glfw.init()
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(GLFW_CONSTANTS.GLFW_CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_PROFILE, 
        GLFW_CONSTANTS.GLFW_OPENGL_CORE_PROFILE
    )
    glfw.window_hint(
        GLFW_CONSTANTS.GLFW_OPENGL_FORWARD_COMPAT, 
        GLFW_CONSTANTS.GLFW_TRUE
    )
    glfw.window_hint(GLFW_CONSTANTS.GLFW_DOUBLEBUFFER, GL_FALSE)

    window = glfw.create_window(SCREEN_WIDTH, SCREEN_HEIGHT, "My Game", None, None)
    glfw.make_context_current(window)
    glfw.set_input_mode(
        window, 
        GLFW_CONSTANTS.GLFW_CURSOR,
        GLFW_CONSTANTS.GLFW_CURSOR_HIDDEN
    )

    return window

class SimpleComponent:

    def __init__(self, position, eulers):

        self.position = np.array(position, dtype=np.float32)
        self.eulers = np.array(eulers, dtype=np.float32)

class Light:

    def __init__(self, position, color, strength):

        self.position = np.array(position, dtype=np.float32)
        self.color = np.array(color, dtype=np.float32)
        self.strength = strength

class Camera:

    def __init__(self, position):

        self.position = np.array(position, dtype=np.float32)
        self.theta = 0
        self.phi = 0
        self.update_vectors()

    def update_vectors(self):

        self.forwards = np.array(
            [
                np.cos(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.theta)) * np.cos(np.deg2rad(self.phi)),
                np.sin(np.deg2rad(self.phi))
            ]
        )

        globalUp = np.array([0, 0, 1], dtype = np.float32)

        self.right = np.cross(self.forwards, globalUp)

        self.up = np.cross(self.right, self.forwards)

class Scene:

    def __init__(self):

        self.cubes = [
            SimpleComponent(
                position = [
                    np.random.uniform(low=0, high=3.0) + 6,
                    np.random.uniform(low=-5.0, high=5.0),
                    np.random.uniform(low=-3.0, high=1.0)
                ],
                eulers = [
                    np.random.uniform(low=0, high=359),
                    np.random.uniform(low=0, high=359),
                    np.random.uniform(low=0, high=359)
                    ]
            )
            for i in range(5)
        ]

        self.medkits = [
            SimpleComponent(
                position = [3,0,0.5],
                eulers = [0,0,0]
            )
        ]

        self.lights = [
            Light(
                position = [
                    np.random.uniform(low=2.0, high=9.0),
                    np.random.uniform(low=-2.0, high=2.0),
                    np.random.uniform(low=2.0, high=4.0)
                ],
                color = [
                    np.random.uniform(low=0.0, high=0.9),
                    np.random.uniform(low=0.0, high=1.0),
                    np.random.uniform(low=0.0, high=1.0)
                ],
                strength = 3
            )
            for _ in range(8)
        ]

        self.player = Camera(position = [0,0,2])

    def update(self, rate):

        for cube in self.cubes:
            cube.eulers[1] += 0.25 * rate
            if cube.eulers[1] > 360:
                cube.eulers[1] -= 360
        

    def move_player(self, dPos):

        # dPos = np.array(dPos, dtype = np.float32)
        self.player.position += dPos
    
    def spin_player(self, dTheta, dPhi):

        self.player.theta += dTheta
        if self.player.theta > 360:
            self.player.theta -= 360
        elif self.player.theta < 0:
            self.player.theta += 360

        self.player.phi = min(
            89, max(-89, self.player.phi + dPhi)
        )

        self.player.update_vectors()

class App:

    def __init__(self, window):

        self.window = window
        self.renderer = GraphicsEngine()
        self.scene = Scene()
        
        self.lastTime = glfw.get_time()
        self.currentTime = 0
        self.numFrames = 0
        self.frameTime = 0

        self.walk_offset_lookup = {
            1: 0,
            2: 90,
            3: 45,
            4: 180,
            6: 135,
            7: 90,
            8: 270,
            9: 315,
            11: 0,
            12: 225,
            13: 270,
            14: 180
        }

        self.mainLoop()

    def mainLoop(self):

        running = True
        while running:
            # check events
            
            if glfw.window_should_close(self.window) \
                or glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_ESCAPE) == GLFW_CONSTANTS.GLFW_PRESS:

                running = False

            self.handleKeys()
            self.handleMouse()


            glfw.poll_events()

            self.scene.update(self.frameTime / 16.7)

            self.renderer.render(self.scene)

            # timing
            self.caculateFramerate()
        self.quit()
    
    def handleKeys(self):

        """
        w: 1 -> 0 degrees
        a: 2 -> 90 degrees
        w & a: 3 -> 45 degrees
        s: 4 -> 180 degrees
        w & s: 5 -> x
        a & s: 6 -> 135 degrees
        w & a & s: -> 90 degrees
        d: 8 -> 270 degrees
        w & d: 9 -> 315 degrees
        a & d: 10 -> x
        w & a & d: 11 -> 0 degrees
        s & d: 12 -> 225 degrees
        w & s & d: 13 -> 270 degrees
        a & s & d: 14 -> 180 degrees
        w & a & s & d: 15 -> x
        """
        combo = 0
        directionModifier = 0

        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_W) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 1
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_A) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 2
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_S) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 4
        if glfw.get_key(self.window, GLFW_CONSTANTS.GLFW_KEY_D) == GLFW_CONSTANTS.GLFW_PRESS:
            combo += 8

        if combo in self.walk_offset_lookup:
            directionModifier = self.walk_offset_lookup[combo]
            dPos = [
                0.1 * self.frameTime / 16.7 * np.cos(np.deg2rad(self.scene.player.theta + directionModifier)),
                0.1 * self.frameTime / 16.7 * np.sin(np.deg2rad(self.scene.player.theta + directionModifier)),
                0
            ]
            self.scene.move_player(dPos)

    def handleMouse(self):

        x, y = glfw.get_cursor_pos(self.window)
        rate = self.frameTime / 16.7
        dTheta = rate * (SCREEN_WIDTH/2 - x)
        dPhi = rate * (SCREEN_HEIGHT/2 - y)
        self.scene.spin_player(dTheta, dPhi)
        glfw.set_cursor_pos(self.window, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

    def caculateFramerate(self):

        self.currentTime = glfw.get_time()
        delta = self.currentTime - self.lastTime
        if delta >= 1:
            framerate = max(1, self.numFrames // delta)
            glfw.set_window_title(self.window, f"Running at {framerate} fps.")
            self.lastTime = self.currentTime
            self.numFrames = -1
            self.frameTime = 1000/max(1,framerate)
        self.numFrames += 1

    def quit(self):
        self.renderer.destroy()

class GraphicsEngine:
    
    def __init__(self):

        # initialize opengl
        glClearColor(0, 0, 0, 1)
        self.shader = self.createShader(r"shaders\vertex.txt", r"shaders\fragment.txt")
        glUseProgram(self.shader)
        glUniform1i(glGetUniformLocation(self.shader, "imageTexture"), 0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)

        # glEnable(GL_CULL_FACE) # 뒷면 안보이게 하기
        # glCullFace(GL_BACK)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.wood_texture = Material(r"gfx\wood.jpeg")
        self.cube_mesh = Mesh(r"models\cube.obj")

        self.medkit_texture = Material(r"gfx\medkit.png")
        self.medkit_billboard = BillBoard(w=0.6, h=0.5)

        self.light_texture = Material(r"gfx\greenlight.png")
        self.light_billboard = BillBoard(w=0.2, h=0.1)

        projection_transform = pyrr.matrix44.create_perspective_projection(
            fovy = 45, aspect = SCREEN_WIDTH/SCREEN_HEIGHT,
            near = 0.1, far = 50, dtype=np.float32
        )
        glUniformMatrix4fv(
            glGetUniformLocation(self.shader, "projection"),
            1, GL_FALSE, projection_transform
        )
        self.modelMatrixLocation = glGetUniformLocation(self.shader, "model")
        self.viewMatrixLocation = glGetUniformLocation(self.shader, "view")
        self.lightLocation = {
            "position": [
                glGetUniformLocation(self.shader, f"Lights[{i}].position")
                for i in range(8)
            ],
            "color": [
                glGetUniformLocation(self.shader, f"Lights[{i}].color")
                for i in range(8)
            ],
            "strength": [
                glGetUniformLocation(self.shader, f"Lights[{i}].strength")
                for i in range(8)
            ]
        }
        self.cameraPosLoc = glGetUniformLocation(self.shader, "cameraPosition")
        self.tintLoc = glGetUniformLocation(self.shader, "tint")

    def createShader(self, vertexFilepath, fragmentFIlepath):

        with open(vertexFilepath, 'r') as f:
            vertex_src = f.readlines()

        with open(fragmentFIlepath, 'r') as f:
            fragment_src = f.readlines()

        shader = compileProgram(
            compileShader(vertex_src, GL_VERTEX_SHADER),
            compileShader(fragment_src, GL_FRAGMENT_SHADER)
        )

        return shader

    def render(self, scene):

        # refresh screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)
        

        view_transform = pyrr.matrix44.create_look_at(
            eye = scene.player.position, 
            target = scene.player.position + scene.player.forwards,
            up = scene.player.up, dtype=np.float32
        )
        glUniformMatrix4fv(self.viewMatrixLocation, 1, GL_FALSE, view_transform)

        for i,light in enumerate(scene.lights):
            glUniform3fv(self.lightLocation["position"][i], 1, light.position)
            glUniform3fv(self.lightLocation["color"][i], 1, light.color)
            glUniform1f(self.lightLocation["strength"][i], light.strength)

        glUniform3fv(self.cameraPosLoc, 1, scene.player.position)

        for light in scene.lights:

            self.light_texture.use()
            glUniform3fv(self.tintLoc, 1, light.color)

            directionFromPlayer = light.position - scene.player.position
            angle1 = np.arctan2(-directionFromPlayer[1], directionFromPlayer[0])
            dist2d = math.sqrt(directionFromPlayer[0]**2 + directionFromPlayer[1]**2)
            angle2 = np.arctan2(directionFromPlayer[2], dist2d)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_y_rotation(theta = angle2, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_z_rotation(theta = angle1, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_translation(vec = light.position, dtype=np.float32)
            )

            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            glBindVertexArray(self.light_billboard.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.light_billboard.vertex_count)

        glUniform3fv(self.tintLoc, 1, np.array([1,1,1],dtype=np.float32))

        for cube in scene.cubes:
            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_eulers(
                    eulers=np.radians(cube.eulers),
                    dtype=np.float32
                )
            )
            model_transform = pyrr.matrix44.multiply(
                m1=model_transform,
                m2=pyrr.matrix44.create_from_translation(
                    vec=np.array(cube.position),
                    dtype=np.float32
                )
            )
            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            self.wood_texture.use()
            glBindVertexArray(self.cube_mesh.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.cube_mesh.vertex_count)

        for medkit in scene.medkits:

            self.medkit_texture.use()

            directionFromPlayer = medkit.position - scene.player.position
            angle1 = np.arctan2(-directionFromPlayer[1], directionFromPlayer[0])
            dist2d = math.sqrt(directionFromPlayer[0]**2 + directionFromPlayer[1]**2)
            angle2 = np.arctan2(directionFromPlayer[2], dist2d)

            model_transform = pyrr.matrix44.create_identity(dtype=np.float32)
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_y_rotation(theta = angle2, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_z_rotation(theta = angle1, dtype=np.float32)
            )
            model_transform = pyrr.matrix44.multiply(
                model_transform,
                pyrr.matrix44.create_from_translation(vec = medkit.position, dtype=np.float32)
            )

            glUniformMatrix4fv(self.modelMatrixLocation, 1, GL_FALSE, model_transform)
            glBindVertexArray(self.medkit_billboard.vao)
            glDrawArrays(GL_TRIANGLES, 0, self.medkit_billboard.vertex_count)


        glFlush()

    def destroy(self):

        self.cube_mesh.destroy()
        self.wood_texture.destroy()
        self.medkit_billboard.destroy()
        self.medkit_texture.destroy()
        self.light_billboard.destroy()
        self.light_texture.destroy()
        glDeleteProgram(self.shader)

class Mesh:

    def __init__(self, filepath):
        
        #x, y, z, s, t, nx, ny, nz
        self.vertices = self.loadMesh(filepath)

        self.vertex_count = len(self.vertices) // 8

        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        # texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        # normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))
        
    def loadMesh(self, filepath):
        
        vertices = []

        # raw, unassembled data
        v = [] 
        vt = []
        vn = []

        with open(filepath, 'r') as f:
            line = f.readline()
            while line:
                firstSpace = line.find(" ")
                flag = line[0:firstSpace]

                if flag == "v":
                    line = line.replace("v", "")
                    # [x, y, z]
                    l = list(map(float,line.split()))
                    v.append(l)
                elif flag == "vt":
                    line = line.replace("vt", "")
                    # [s, t]
                    l = list(map(float,line.split()))
                    vt.append(l)
                elif flag == "vn":
                    line = line.replace("vn", "")
                    # [nx, ny, nz]
                    l = list(map(float,line.split()))
                    vn.append(l)
                elif flag == "f":
                    # face, three or more vertices in v/vt/vn form
                    line = line.replace("f", "")
                    line = line.replace('\n',"")
                    # [../../.., ../../.., ../../..]
                    line = line.split()
                    faceVertices = []
                    faceTextures = []
                    faceNormals = []
                    for vertex in line:
                        # vertex = v/vt/vn
                        # [v,vt,vn]
                        l = vertex.split("/")
                        
                        faceVertices.append(v[int(l[0])-1])
                        faceTextures.append(vt[int(l[1])-1])
                        faceNormals.append(vn[int(l[2])-1])
                    # [0,1,2,3] -> [0,1,2,0,2,3]
                    triangles_in_face = len(line) - 2
                    vertex_order = []
                    for i in range(triangles_in_face):
                        vertex_order.append(0)
                        vertex_order.append(i + 1)
                        vertex_order.append(i + 2)
                    for i in vertex_order:
                        for x in faceVertices[i]:
                            vertices.append(x)
                        for x in faceTextures[i]:
                            vertices.append(x)
                        for x in faceNormals[i]:
                            vertices.append(x)
                line = f.readline()
        return vertices

    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vao,))  

class Material:

    def __init__(self, filepath):

        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(filepath, mode = 'r') as image:
            image_width, image_height = image.size
            image = image.convert("RGBA")
            image_data = bytes(image.tobytes())
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glGenerateMipmap(GL_TEXTURE_2D)

    def use(self):
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def destroy(self):

        glDeleteTextures(1, (self.texture,))

class BillBoard:

    def __init__(self, w , h):

        # x,y,z, s,t, normal
        self.vertices = (
            0, -w/2,  h/2, 0, 0, -1, 0, 0,
            0, -w/2, -h/2, 0, 1, -1, 0, 0,
            0,  w/2, -h/2, 1, 1, -1, 0, 0,

            0, -w/2,  h/2, 0, 0, -1, 0, 0,
            0,  w/2, -h/2, 1, 1, -1, 0, 0,
            0,  w/2,  h/2, 1, 0, -1, 0, 0
        )

        self.vertex_count = 6

        self.vertices = np.array(self.vertices, dtype=np.float32)

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        # position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        # texture
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        # normal
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(20))

    def destroy(self):

        glDeleteVertexArrays(1, (self.vao,))
        glDeleteBuffers(1, (self.vao,)) 

if __name__ == "__main__":
    print("start")
    window = initialize_glfw()
    myApp = App(window)