import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw


vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )

colors = np.random.randn(12, 3)

# colors = (
#     (1,0,0),
#     (0,1,0),
#     (0,0,1),
#     (0,1,0),
#     (1,1,1),
#     (0,1,1),
#     (1,0,0),
#     (0,1,0),
#     (0,0,1),
#     (1,0,0),
#     (1,1,1),
#     (0,1,1),
#     )

surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )

def Cube():
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            glColor3fv(colors[x])
            glVertex3fv(vertices[vertex])
    glEnd()

    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()


def main():

    height, width = 400, 400
    previous = np.zeros((height, width, 3), dtype=np.uint8)

    display = (height, width)

    if not glfw.init():
        return
    glfw.window_hint(glfw.VISIBLE, False)
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(width, height, "hidden window", None, None)
    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)


    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    # glTranslatef(0, 0, -5)
    glTranslatef(0, 0, -10)
    # glRotatef(25, 2, 1, 0)

    rspeed = np.random.randn(3) * 0.2

    z = -0.1
    while True:
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         pygame.quit()
        #         quit()
        #
        # if event.type == pygame.KEYDOWN:
        #     if event.key == pygame.K_LEFT:
        #         glTranslatef(-0.5, 0, 0)
        #     if event.key == pygame.K_RIGHT:
        #         glTranslatef(0.5, 0, 0)
        #
        #     if event.key == pygame.K_UP:
        #         glTranslatef(0, 1, 0)
        #     if event.key == pygame.K_DOWN:
        #         glTranslatef(0, -1, 0)
        #
        # if event.type == pygame.MOUSEBUTTONDOWN:
        #     if event.button == 1:
        #         glTranslatef(0, 0, 0.3)
        #
        #     if event.button == 3:
        #         glTranslatef(0, 0, -0.3)

        rspeed = rspeed * 0.99 + np.random.randn(3) * 0.2
        #R = cv2.Rodrigues(rspeed)[0]

        x = glGetDoublev(GL_MODELVIEW_MATRIX)
        camera_x = x[3][0]
        camera_y = x[3][1]
        camera_z = x[3][2]
        #glTranslatef(camera_x, camera_y, camera_z)
        glRotatef(rspeed[0], 3, 1, 1)
        glRotatef(rspeed[1], 0, 1, 0)
        glRotatef(rspeed[2], 0, 0, 1)

        #glTranslatef(-camera_x, -camera_y, -camera_z)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        Cube()


        #Grab Buffer with numpy & display it using OpenCV
        data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, outputType=None)
        im = np.fromstring(data, dtype=np.uint8, count=-1).reshape(height, width, 3)[:,:,::-1]
        diff = im - previous
        previous[...] = im
        cv2.imshow('im', im)
        cv2.imshow('diff', diff)
        cv2.waitKey(5)

if __name__ == '__main__':
    main()