/********************************************************************************************************
This code is owned by
    * Ashik KN
    * Deepthi R
    * Kavya V
    * KVS Srikrishna Chaitanya
*********************************************************************************************************
*/


//This is the Message Board program using  sockets to communicate with the visual surveillance system. It displays all warnings and messages from the VSS.

#include<iostream>

#include<sys/types.h>
#include<sys/socket.h>
#include<sys/stat.h>
#include<unistd.h>
#include<stdlib.h>
#include<fcntl.h>

#include<netinet/in.h>
#include<arpa/inet.h>

#define BUFFER_SIZE 1024

#define TEST 1
#define STATIONARY 2
#define BAG 3
#define PERSON 4

using namespace std;
int main()
{
    int cont,create_socket,new_socket,fd;
    unsigned int addrlen;
    char received[256];
    struct sockaddr_in address;

    char *buffer = new char[BUFFER_SIZE];
    if ((create_socket = socket(AF_INET,SOCK_STREAM,0)) > 0)  //Create a new socket
         cout<<"\n > Socket Created.... ";

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;	   // This specifies that any form of incomig message must be accepted
        address.sin_port = htons(15000);             //htons() converts unsigned short from host byte order to network byte order

    cout<<"\n > Message Board Online.... ";

    if(bind(create_socket,(struct sockaddr*)&address,sizeof(address))==0)  //Binds the created socket to an address
        cout<<"\n > Binding Socket.... ";


    while(1)
    {
        listen(create_socket,3);  //Listen for connections on a socket. the second argument is queue size

        addrlen = sizeof(struct sockaddr_in);

        new_socket = accept(create_socket,(struct sockaddr *)&address,&addrlen);   //Accept a connection on a socket

        if(new_socket>0)
            cout<<"\n > Connection with Client established. (Client Address= "<<inet_ntoa(address.sin_addr) <<"....)";

        recv(new_socket,received, 255,0);
        cout<<"\n Received Data: "<<received;

        switch(atoi(received))
        {
        case 0:
            break;

        case TEST:
            cout<<"\n >> Connection with Visual Surveillance System tested..... OK"<<endl;
            break;

        case STATIONARY:
            cout<<"\n >> Stationary Object Detected...."<<endl;
            break;

        case BAG:
            cout<<"\n >> WARNING: Unattended Bag Detected!!!!"<<endl;
            system("play -q -V0  alert_clipped.mp3 &");
            break;

        case PERSON:
            cout<<"\n >> WARNING: Loitering Person Detected!!!!!"<<endl;
            break;

        default:
            break;
        }

        //memset(received,'0',255);
    }


    cout<<endl<<endl;
    close(new_socket);
    return close(create_socket);

}
