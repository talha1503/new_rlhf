SHELL:=cmd.exe
annDir:=${CURDIR}

export PYTHONPATH:=$(annDir)
$(info Setting environment PYTHONPATH=$(shell echo $${PYTHONPATH}))

detectedOS:=$(shell uname)
$(info Detected OS: $(detectedOS))

ifeq ($(detectedOS),Linux)
	openCmd:=xdg-open
else
	openCmd:=start
endif


test:
	echo Hello World

begin_session:
	bash scripts\frontend_end.bat || echo "The specified ports are already closed." && \
	pip3 install flask && \
	bash scripts\frontend_start.bat && \
	bash scripts\app_start.bat && \
	timeout 2 && \
	$(openCmd) http://127.0.0.1:5000/

end_session:
	bash app\frontend_end.bat || echo "The specified ports are already closed."

