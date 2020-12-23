/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mycompany.serverutilities;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

/**
 *
 * @author thoma
 */
public class Server {
       
    public Server(int portNumber) throws IOException {        
        /*
         * Create a ServerSocket object to listen on Port <portNumber>.
         * ServerSocket is a java.net class that provides
         * a system-independent implementation of a client/server
         * socket connection. The constructor for ServerSocket
         * throws an exception if it can't listen on the specified port.
         */
        this.serverSocket = new ServerSocket(portNumber);
        
        /* 
         * TODO: Have Server constructor throw a server-specific exception
         * instead of IOException.
         */
    }
    
    /*
     * Listens for messages from clients and transmits responses
     * to appropriate clients.
     */
    public void listenAndRespond() {
        
        String inputLine;
        while (true) {
        
            // Handles IOExceptions from serverSocket.accept().
            try {

                /*
                 * Creates a Socket object which is bound to Port <portNumber>
                 * and has its remote address and remote port set to the address
                 * and port of a client that has requested and successfully
                 * established a connection. The server can communicate with the
                 * client over this new Socket and continue to listen for client
                 * connection requests on the original ServerSocket. This
                 * particular program doesn't listen for more client connection
                 * requests.
                 */
                Socket clientSocket = this.serverSocket.accept();
                
                /*
                 * Create a BufferedReader on the clientSocket's input stream.   
                 */
                BufferedReader bufferedReader = new BufferedReader(
                    new InputStreamReader(clientSocket.getInputStream()));
                
                /*
                 * Create a PrintWriter on the clientSocket's output stream.   
                 */
                PrintWriter printWriter =
                        new PrintWriter(clientSocket.getOutputStream(), true);
                
                /*
                 * Infinite loop to listen for messages from clients.
                 */
                while ((inputLine = bufferedReader.readLine()) != null) {

                    // Echo back inputLine from client socket's input stream
                    // to client socket's output stream.
                    printWriter.println(inputLine);

                    /*
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
                
            }
            catch (IOException e) {
                // Write IOException message to a log file using Logger.
            }
        }
    }
    
    private final ServerSocket serverSocket;
}
