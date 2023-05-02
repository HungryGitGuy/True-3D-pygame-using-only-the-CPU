# True-3D-pygame-using-only-the-CPU
This was a little translation project from One Lone Coders Code it yourself 3d graphics series. I tried to keep it as readable as possible but it's my first big translation project and it's gonna get messy sometimes.  

The only required libraries are pygame and numba (at least the first version of the project requires them) and it shouldnt be too hard to remove the requirement for numba as I don't think I actually used it. 

I'm aware the camera is a bit broken, I'll try to fix it in the future.

Once you have the required libraries installed, just give it a .obj file to render and run (look at the code, just above the mainloop there's  some sample code in the on_run() function for importing a file)

I'll add instructions as to how to convert a regular blender file into a compatible format, for now you can just use this:

# Blender v3.0.0 OBJ File: ''
# www.blender.org
o Cube_Cube.001
v -3.933865 0.101005 3.703212
v -3.933865 -0.101005 3.703212
v -3.933865 0.101005 -3.703212
v -3.933865 -0.101005 -3.703212
v 3.933865 0.101005 3.703212
v 3.933865 -0.101005 3.703212
v 3.933865 0.101005 -3.703212
v 3.933865 -0.101005 -3.703212
s off
f 2 3 1
f 4 7 3
f 8 5 7
f 6 1 5
f 7 1 3
f 4 6 8
f 2 4 3
f 4 8 7
f 8 6 5
f 6 2 1
f 7 5 1
f 4 2 6
o Cylinder
v 1.911590 0.100000 -2.341112
v 1.911590 2.100000 -2.341112
v 1.987675 0.100000 -2.285834
v 1.987675 2.100000 -2.285834
v 1.958613 0.100000 -2.196391
v 1.958613 2.100000 -2.196391
v 1.864567 0.100000 -2.196391
v 1.864567 2.100000 -2.196391
v 1.835505 0.100000 -2.285834
v 1.835505 2.100000 -2.285834
s off
f 10 11 9
f 12 13 11
f 14 15 13
f 18 16 14
f 16 17 15
f 18 9 17
f 11 15 17
f 10 12 11
f 12 14 13
f 14 16 15
f 14 12 10
f 10 18 14
f 16 18 17
f 18 10 9
f 17 9 11
f 11 13 15
o Cube.001_Cube.002
v 1.840639 1.701495 -2.289564
v 1.583514 2.215569 -2.679421
v 1.904668 1.759362 -2.336083
v 1.613347 2.242531 -2.701096
v 1.840639 1.810552 -2.289564
v 1.583514 2.266382 -2.679421
v 1.904668 1.868419 -2.336083
v 1.613347 2.293344 -2.701096
v 1.911590 1.961886 -2.261112
v 2.635190 2.343234 -1.735392
v 1.635205 2.343234 -1.410472
v 1.017165 2.343234 -2.261112
v 1.635205 2.343234 -3.111753
v 2.635190 2.343234 -2.786832
v 2.187975 2.960271 -1.410472
v 1.187990 2.960271 -1.735392
v 1.187990 2.960271 -2.786832
v 2.187975 2.960271 -3.111752
v 2.806015 2.960271 -2.261112
v 1.911590 3.341619 -2.261112
s off
f 20 21 19
f 22 25 21
f 25 24 23
f 23 20 19
f 21 23 19
f 22 24 26
f 27 28 29
f 28 27 32
f 27 29 30
f 27 30 31
f 27 31 32
f 28 32 37
f 29 28 33
f 30 29 34
f 31 30 35
f 32 31 36
f 28 37 33
f 29 33 34
f 30 34 35
f 31 35 36
f 32 36 37
f 33 37 38
f 34 33 38
f 35 34 38
f 36 35 38
f 37 36 38
f 20 22 21
f 22 26 25
f 25 26 24
f 23 24 20
f 21 25 23
f 22 20 24
