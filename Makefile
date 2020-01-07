SCRIPT := aistat19

default: run
	
run:
	matlab -nodisplay -nosplash -nodesktop -r "$(SCRIPT);quit"
