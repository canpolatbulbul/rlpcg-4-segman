import sys
import robotic as ry

if __name__ == "__main__":
    filename = sys.argv[1]

    C = ry.Config()
    C.addFile(filename)
    C.view(True)
    C.view_close()
