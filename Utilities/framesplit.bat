for %%A IN (*.avi) DO (
	mkdir %%~nA
	ffmpeg -i %%~A %%~nA/file%%05d.bmp
	move %%A %%~nA
)