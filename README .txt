 - Compile and execute message board first in one terminal
 	g++ message_board.cpp  -o mb
	./mb
 - Compile and execute the main program in another terminal
	g++ visual_surveillance_system.cpp -o vss `pkg-config --cflags --libs opencv`
	./vss		
