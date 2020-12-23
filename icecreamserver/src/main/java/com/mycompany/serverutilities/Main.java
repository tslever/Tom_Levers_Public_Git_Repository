
/*
 * Allows executable to find class Main.
 */
package com.mycompany.serverutilities;

/*
 * Allows use of isNumeric.
 */
import org.apache.commons.lang.StringUtils;

/*
 * Allows use of classes ServerSocket and Socket.
 */
import java.net.ServerSocket;
import java.net.Socket;

/*
 * Allows use of classes IOException, PrintWriter, BufferedReader, and
 * InputStreamReader.
 */
import java.io.IOException;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 * Defines Main class that encapsulates static method main,
 * the entry point of this application.
 * This application instantiates a server that listens for messages.
 * @version 0.0
 * @author Tom Lever
 */
public class Main {
    
    /**
     * Defines the main method of this application.
     * @param args
     */
    public static void main(String[] args) {
        
        System.out.println("Started method main.");
        
        // Check that portNumber is in command-line arguments.
        if ((args.length != 1) || (!StringUtils.isNumeric(args[0]))) {
            System.err.println(
                    "Usage error. Usage: java EchoServer <portNumber>");
            System.exit(1);
        }
        
        // Parse portNumber from command-line arguments.
        int portNumber = Integer.parseInt(args[0]);
        System.out.printf("User-specified Port number: %d.\n", portNumber);
        
        
        /*
         * TODO: Wrap the following try / catch block to class Server.
         * Pass portNumber to Server's constructor.
         */
        
        /*
         * Try creating resources for reading messages from a client,
         * waiting for messages from client,
         * and returning messages to the client.
         */
        try (
            /*
             * Create a ServerSocket object to listen on Port <portNumber>.
             * ServerSocket is a java.net class that provides
             * a system-independent implementation of a client/server
             * socket connection. The constructor for ServerSocket
             * throws an exception if it can't listen on the specified port.
             */
            ServerSocket serverSocket = new ServerSocket(portNumber);
                
            /*
             * Create a Socket object which is bound to Port <portNumber>
             * and has its remote address and remote port set to the address
             * and port of a client that has requested and successfully
             * established a connection. The server can communicate with the
             * client over this new Socket and continue to listen for client
             * connection requests on the original ServerSocket. This
             * particular program doesn't listen for more client connection
             * requests.
             */
            Socket clientSocket = serverSocket.accept();
                
            /*
             * Create a PrintWriter on the clientSocket's output stream.   
             */
            PrintWriter printWriter =
                    new PrintWriter(clientSocket.getOutputStream(), true);
                
            /*
             * Create a BufferedReader on the clientSocket's input stream.   
             */
            BufferedReader bufferedReader = new BufferedReader(
                new InputStreamReader(clientSocket.getInputStream()));
        ) {
            
            /*
             * Wait for a message from a client and
             * return the client's message to the client.
             */
            String inputLine;
            while ((inputLine = bufferedReader.readLine()) != null) {
                printWriter.println(inputLine);
            }
            
            /*
             * TODO: Instead of having message interface echo messages
             * immediately, have message interface store received messages
             * in a queue.
             * 
             * Parse messages into information.
             *
             * Create ice-cream flavors based on information.
             *
             * Perform CRUD operations.
             *
             * Create responses.
             *
             * Send messages.
             *
             * Remove input messages corresponding to responses from queue.
             */
        }
        
        /*
         * If creating resources for reading messages from a client failed,
         * receiving messages from client failed,
         * or returning messages to client failed,
         * process this error as an IOException and
         * print a message to console.
         */
        catch (IOException e) {
            System.out.println(
                    "Exception caught when trying to listen on port " +
                    portNumber +
                    " or listening for a connection.");
            System.out.println(e.getMessage());
        }
    }
}
