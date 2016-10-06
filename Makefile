default:
	nvcc -lglfw3 -Xlinker -framework,Cocoa,-framework,OpenGL,-framework,IOKit,-framework,CoreVideo  main.cu